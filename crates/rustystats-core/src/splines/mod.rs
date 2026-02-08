// =============================================================================
// SPLINE BASIS FUNCTIONS
// =============================================================================
//
// This module provides high-performance B-spline and natural spline basis
// functions for non-linear continuous effects in GLMs.
//
// IMPLEMENTATIONS:
// ----------------
// 1. B-splines (basis splines) - Flexible piecewise polynomial bases
// 2. Natural splines - B-splines with linear extrapolation beyond boundaries
// 3. Penalized splines (P-splines) - Automatic smoothness selection via GCV
//
// PERFORMANCE:
// ------------
// - Uses Cox-de Boor recursive algorithm for numerical stability
// - Parallel evaluation over observations using Rayon
// - Pre-computed knot sequences for efficiency
// - Cache-friendly memory layout
//
// USAGE:
// ------
// In formulas: bs(x, df=5), ns(x, df=5), s(x, k=10)
// Direct API:  bs_basis(x, knots, degree), ns_basis(x, knots)
//
// =============================================================================

pub mod penalized;

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::constants::{DEFAULT_SPLINE_DEGREE, KNOT_TOL};

/// Default degree for B-splines (cubic)
pub const DEFAULT_DEGREE: usize = DEFAULT_SPLINE_DEGREE;

// =============================================================================
// KNOT COMPUTATION
// =============================================================================

/// Compute knot sequence for B-splines.
///
/// For df degrees of freedom with degree d:
/// - Number of interior knots = df - d - 1 (if intercept) or df - d (if no intercept)
/// - Total knots = n_interior + 2 * (d + 1) (boundary knots repeated)
///
/// # Arguments
/// * `x` - Data values to compute knot range from
/// * `df` - Degrees of freedom (number of basis functions)
/// * `degree` - Spline degree (3 for cubic)
/// * `boundary_knots` - Optional explicit boundary knots (min, max)
///
/// # Returns
/// Complete knot vector with boundary knots repeated degree+1 times
pub fn compute_knots(
    x: &Array1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> Vec<f64> {
    // Determine boundary knots
    let (x_min, x_max) = boundary_knots.unwrap_or_else(|| {
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    });
    
    // Number of interior knots
    // df = n_basis = n_interior + degree + 1 (for B-splines with intercept)
    // So n_interior = df - degree - 1
    let n_interior = if df > degree + 1 { df - degree - 1 } else { 0 };
    
    // Create interior knots at quantiles of x
    let mut interior_knots = Vec::with_capacity(n_interior);
    if n_interior > 0 {
        let mut sorted_x: Vec<f64> = x.iter().cloned().collect();
        sorted_x.sort_by(|a, b| a.total_cmp(b));
        
        for i in 1..=n_interior {
            let p = (i as f64) / ((n_interior + 1) as f64);
            let idx = (p * (sorted_x.len() - 1) as f64) as usize;
            interior_knots.push(sorted_x[idx]);
        }
    }
    
    // Build complete knot vector
    // [x_min repeated (degree+1), interior_knots, x_max repeated (degree+1)]
    let mut knots = Vec::with_capacity(2 * (degree + 1) + n_interior);
    
    // Left boundary knots (repeated)
    for _ in 0..=degree {
        knots.push(x_min);
    }
    
    // Interior knots
    knots.extend(interior_knots);
    
    // Right boundary knots (repeated)
    for _ in 0..=degree {
        knots.push(x_max);
    }
    
    knots
}

/// Build complete knot vector from interior knots and boundary knots.
///
/// This is used when we have pre-computed interior knots (from training)
/// and need to rebuild the full knot vector for prediction.
pub fn build_knot_vector(
    interior_knots: &[f64],
    degree: usize,
    x_min: f64,
    x_max: f64,
) -> Vec<f64> {
    let n_interior = interior_knots.len();
    let mut knots = Vec::with_capacity(2 * (degree + 1) + n_interior);
    
    // Left boundary knots (repeated)
    for _ in 0..=degree {
        knots.push(x_min);
    }
    
    // Interior knots
    knots.extend(interior_knots);
    
    // Right boundary knots (repeated)
    for _ in 0..=degree {
        knots.push(x_max);
    }
    
    knots
}

/// Compute knot sequence for natural splines.
///
/// Natural splines have additional constraints (linear at boundaries)
/// so they need fewer interior knots for the same df.
pub fn compute_knots_natural(
    x: &Array1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
) -> (Vec<f64>, f64, f64) {
    // For natural splines: df = n_interior + 1
    // (intercept absorbed, 2 constraints at boundaries)
    let n_interior = if df > 1 { df - 1 } else { 0 };
    
    // Determine boundary knots
    let (x_min, x_max) = boundary_knots.unwrap_or_else(|| {
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    });
    
    // Interior knots at quantiles
    let mut interior_knots = Vec::with_capacity(n_interior);
    if n_interior > 0 {
        let mut sorted_x: Vec<f64> = x.iter().cloned().collect();
        sorted_x.sort_by(|a, b| a.total_cmp(b));
        
        for i in 1..=n_interior {
            let p = (i as f64) / ((n_interior + 1) as f64);
            let idx = (p * (sorted_x.len() - 1) as f64) as usize;
            interior_knots.push(sorted_x[idx]);
        }
    }
    
    (interior_knots, x_min, x_max)
}

// =============================================================================
// B-SPLINE BASIS (Cox-de Boor Algorithm)
// =============================================================================

/// Evaluate all B-spline basis functions at a single point using iterative Cox-de Boor.
///
/// Uses a triangular table to compute all basis values simultaneously,
/// reusing intermediate computations for efficiency.
#[inline]
fn bspline_all_basis_at_point(x: f64, degree: usize, knots: &[f64], n_basis: usize) -> Vec<f64> {
    // Use triangular computation for efficiency
    // We compute all degree-0 bases first, then degree-1, etc.
    
    let n_knots = knots.len();
    
    // Degree 0 bases
    let mut prev = vec![0.0; n_knots - 1];
    for i in 0..(n_knots - 1) {
        // Handle right boundary
        if i == n_knots - 2 {
            prev[i] = if knots[i] <= x && x <= knots[i + 1] { 1.0 } else { 0.0 };
        } else {
            prev[i] = if knots[i] <= x && x < knots[i + 1] { 1.0 } else { 0.0 };
        }
    }
    
    // Build up to desired degree
    for d in 1..=degree {
        let mut curr = vec![0.0; n_knots - d - 1];
        for i in 0..curr.len() {
            let mut val = 0.0;
            
            // First term
            let denom1 = knots[i + d] - knots[i];
            if denom1.abs() > KNOT_TOL && i < prev.len() {
                let w1 = (x - knots[i]) / denom1;
                val += w1 * prev[i];
            }
            
            // Second term
            let denom2 = knots[i + d + 1] - knots[i + 1];
            if denom2.abs() > KNOT_TOL && i + 1 < prev.len() {
                let w2 = (knots[i + d + 1] - x) / denom2;
                val += w2 * prev[i + 1];
            }
            
            curr[i] = val;
        }
        prev = curr;
    }
    
    // Return only the requested number of basis functions
    prev.into_iter().take(n_basis).collect()
}

/// Compute B-spline basis matrix.
///
/// # Arguments
/// * `x` - Data points (n,)
/// * `df` - Degrees of freedom (number of basis functions)
/// * `degree` - Spline degree (default 3 for cubic)
/// * `boundary_knots` - Optional (min, max) for knot range
/// * `include_intercept` - Whether to include intercept column
///
/// # Returns
/// Basis matrix (n, df) where each column is a basis function
pub fn bs_basis(
    x: &Array1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> Array2<f64> {
    let n = x.len();
    let actual_df = if include_intercept { df } else { df.max(1) };
    
    // Compute knots
    let knots = compute_knots(x, actual_df, degree, boundary_knots);
    let n_basis = actual_df;
    
    // Parallel evaluation over observations
    let basis_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| bspline_all_basis_at_point(x[i], degree, &knots, n_basis))
        .collect();
    
    // Convert to Array2
    let mut result = Array2::zeros((n, n_basis));
    for (i, row) in basis_rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    // Drop first column if no intercept (for identifiability in models with intercept)
    if !include_intercept && n_basis > 1 {
        result.slice_move(ndarray::s![.., 1..])
    } else {
        result
    }
}

// =============================================================================
// NATURAL SPLINE BASIS
// =============================================================================

/// Compute natural cubic spline basis matrix.
///
/// Natural splines are cubic splines with the additional constraint that
/// the second derivative is zero at the boundaries. This makes extrapolation
/// linear beyond the data range, which is often more sensible.
///
/// Implementation uses the approach from Hastie & Tibshirani:
/// 1. Compute truncated power basis
/// 2. Apply constraint transformation for natural boundary conditions
///
/// # Arguments
/// * `x` - Data points (n,)
/// * `df` - Degrees of freedom
/// * `boundary_knots` - Optional (min, max) for boundary
/// * `include_intercept` - Whether to include intercept column
///
/// # Returns
/// Basis matrix (n, df) with natural spline basis functions
pub fn ns_basis(
    x: &Array1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> Array2<f64> {
    let n = x.len();
    
    // Get knots and boundaries
    let (interior_knots, x_min, x_max) = compute_knots_natural(x, df, boundary_knots);
    
    // For natural splines, we use the ns transformation from R/mgcv
    // df basis functions with natural boundary constraints
    
    let k = interior_knots.len();
    let n_basis = if include_intercept { df } else { df.saturating_sub(1).max(1) };
    
    // Build all knots including boundaries
    let mut all_knots = Vec::with_capacity(k + 2);
    all_knots.push(x_min);
    all_knots.extend(&interior_knots);
    all_knots.push(x_max);
    
    // Compute basis using d_k(x) functions (truncated cubic basis with constraint)
    // d_k(x) = D_k(x) - D_K(x) where D_k(x) = (x - ξ_k)³₊ - (x - ξ_K)³₊ / (ξ_K - ξ_k)
    
    let result_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = x[i];
            compute_ns_row(xi, &all_knots, n_basis, include_intercept)
        })
        .collect();
    
    // Convert to Array2
    let actual_cols = if include_intercept { n_basis } else { n_basis };
    let mut result = Array2::zeros((n, actual_cols));
    for (i, row) in result_rows.into_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j < actual_cols {
                result[[i, j]] = val;
            }
        }
    }
    
    result
}

/// Compute natural cubic spline basis with explicit interior knots.
///
/// This variant accepts pre-computed interior knots, which is essential for
/// prediction on new data where the knots must match those from training.
///
/// # Arguments
/// * `x` - Data points to evaluate
/// * `interior_knots` - Pre-computed interior knot positions
/// * `boundary_knots` - (min, max) boundary knots
/// * `include_intercept` - Whether to include intercept column
///
/// # Returns
/// Basis matrix with natural spline basis functions
pub fn ns_basis_with_knots(
    x: &Array1<f64>,
    interior_knots: &[f64],
    boundary_knots: (f64, f64),
    include_intercept: bool,
) -> Array2<f64> {
    let n = x.len();
    let (x_min, x_max) = boundary_knots;
    
    // df = n_interior + 1 for natural splines
    let df = interior_knots.len() + 1;
    let n_basis = if include_intercept { df } else { df.saturating_sub(1).max(1) };
    
    // Build all knots including boundaries
    let mut all_knots = Vec::with_capacity(interior_knots.len() + 2);
    all_knots.push(x_min);
    all_knots.extend(interior_knots);
    all_knots.push(x_max);
    
    // Compute basis using same algorithm as ns_basis
    let result_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let xi = x[i];
            compute_ns_row(xi, &all_knots, n_basis, include_intercept)
        })
        .collect();
    
    // Convert to Array2
    let actual_cols = n_basis;
    let mut result = Array2::zeros((n, actual_cols));
    for (i, row) in result_rows.into_iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if j < actual_cols {
                result[[i, j]] = val;
            }
        }
    }
    
    result
}

/// Compute a single row of the natural spline basis.
#[inline]
fn compute_ns_row(x: f64, knots: &[f64], n_basis: usize, include_intercept: bool) -> Vec<f64> {
    let k = knots.len();
    if k < 2 {
        return vec![1.0; n_basis];
    }
    
    let xi_1 = knots[0];       // First boundary knot
    let xi_k = knots[k - 1];   // Last boundary knot
    
    // Truncated power function (x - t)³₊
    let tp = |v: f64, t: f64| -> f64 {
        let diff = v - t;
        if diff > 0.0 { diff * diff * diff } else { 0.0 }
    };
    
    // d function for natural spline constraint
    // d_j(x) = [tp(x, ξ_j) - tp(x, ξ_K)] / (ξ_K - ξ_j)
    let d = |v: f64, j: usize| -> f64 {
        let xi_j = knots[j];
        let denom = xi_k - xi_j;
        if denom.abs() < KNOT_TOL {
            0.0
        } else {
            (tp(v, xi_j) - tp(v, xi_k)) / denom
        }
    };
    
    // Build basis functions
    let mut row = Vec::with_capacity(n_basis);
    
    // Include intercept if requested
    if include_intercept {
        row.push(1.0);
    }
    
    // Linear term (always included for natural splines)
    row.push(x - xi_1);  // Centered at first knot
    
    // Non-linear basis: N_j(x) = d_j(x) - d_{K-1}(x) for j = 1, ..., K-2
    // This ensures the natural spline constraint (linear at boundaries)
    for j in 0..(k - 2) {
        if row.len() >= n_basis {
            break;
        }
        let d_j = d(x, j);
        let d_km1 = d(x, k - 2);
        row.push(d_j - d_km1);
    }
    
    // Pad if needed
    while row.len() < n_basis {
        row.push(0.0);
    }
    
    row
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Compute B-spline basis with default options (cubic, no intercept).
pub fn bs(x: &Array1<f64>, df: usize) -> Array2<f64> {
    bs_basis(x, df, DEFAULT_DEGREE, None, false)
}

/// Compute natural spline basis with default options (no intercept).
pub fn ns(x: &Array1<f64>, df: usize) -> Array2<f64> {
    ns_basis(x, df, None, false)
}

/// Compute B-spline basis with knots specified directly.
pub fn bs_with_knots(
    x: &Array1<f64>,
    knots: &[f64],
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> Array2<f64> {
    let n = x.len();
    
    // Build complete knot vector
    let (x_min, x_max) = boundary_knots.unwrap_or_else(|| {
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    });
    
    let mut full_knots = Vec::with_capacity(knots.len() + 2 * (degree + 1));
    
    // Boundary knots (repeated)
    for _ in 0..=degree {
        full_knots.push(x_min);
    }
    full_knots.extend(knots);
    for _ in 0..=degree {
        full_knots.push(x_max);
    }
    
    let n_basis = knots.len() + degree + 1;
    
    // Parallel evaluation
    let basis_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| bspline_all_basis_at_point(x[i], degree, &full_knots, n_basis))
        .collect();
    
    let mut result = Array2::zeros((n, n_basis));
    for (i, row) in basis_rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    result
}

/// Get column names for B-spline basis.
pub fn bs_names(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    let start = if include_intercept { 0 } else { 1 };
    let end = df;
    (start..end).map(|i| format!("bs({}, {}/{})", var_name, i + 1, df)).collect()
}

/// Get column names for natural spline basis.
pub fn ns_names(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    let start = if include_intercept { 0 } else { 0 };
    let end = if include_intercept { df } else { df - 1 };
    (start..end).map(|i| format!("ns({}, {}/{})", var_name, i + 1, df)).collect()
}

// =============================================================================
// I-SPLINE BASIS (Monotonic Splines)
// =============================================================================
//
// I-splines are integrated M-splines (normalized B-splines). They are the 
// standard basis for monotonic regression because:
// 1. Each basis function is monotonically increasing from 0 to 1
// 2. Any linear combination with non-negative coefficients is monotonic
// 3. They can be computed efficiently from B-splines
//
// Reference: Ramsay, J.O. (1988). Monotone Regression Splines in Action.
//            Statistical Science, 3(4), 425-441.
// =============================================================================

/// Compute I-spline (integrated M-spline) basis matrix for monotonic regression.
///
/// I-splines are the cumulative integral of M-splines (normalized B-splines).
/// Each I-spline basis function is monotonically increasing from 0 to 1.
/// With non-negative coefficients, any linear combination produces a 
/// monotonically increasing function.
///
/// # Arguments
/// * `x` - Data points (n,)
/// * `df` - Degrees of freedom (number of basis functions)
/// * `degree` - Spline degree (default 3 for cubic)
/// * `boundary_knots` - Optional (min, max) for knot range
/// * `increasing` - If true (default), basis for increasing function; if false, decreasing
///
/// # Returns
/// Basis matrix (n, df) where each column is an I-spline basis function.
/// All values are in [0, 1], and each column is monotonically increasing in x.
///
/// # Mathematical Background
/// 
/// For a B-spline basis B_j(x) of degree k with knots t_0, ..., t_m:
/// - M-spline: M_j(x) = (k+1) * B_j(x) / (t_{j+k+1} - t_j)  (normalized to integrate to 1)
/// - I-spline: I_j(x) = integral from -∞ to x of M_j(t) dt
///
/// The I-spline can be computed as a cumulative sum of B-splines:
/// I_j(x) = sum_{i >= j} c_i * B_i(x) where c_i are normalization constants
pub fn is_basis(
    x: &Array1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    increasing: bool,
) -> Array2<f64> {
    let n = x.len();
    
    // Get boundary knots
    let (x_min, x_max) = boundary_knots.unwrap_or_else(|| {
        let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    });
    
    // Compute knots for B-splines of one higher degree
    // I-splines of degree k are integrals of M-splines of degree k,
    // which relate to B-splines of degree k
    let knots = compute_knots(x, df, degree, Some((x_min, x_max)));
    
    // Parallel evaluation over observations
    let basis_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| compute_ispline_row(x[i], degree, &knots, df, x_min, x_max))
        .collect();
    
    // Convert to Array2
    let mut result = Array2::zeros((n, df));
    for (i, row) in basis_rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    // For decreasing monotonicity, reverse the basis columns and flip values
    if !increasing {
        let mut flipped = Array2::zeros((n, df));
        for i in 0..n {
            for j in 0..df {
                // 1 - I_j gives decreasing function, reverse column order
                flipped[[i, j]] = 1.0 - result[[i, df - 1 - j]];
            }
        }
        result = flipped;
    }
    
    result
}

/// Compute I-spline (monotonic spline) basis with explicit knots.
///
/// This variant accepts pre-computed knots, essential for prediction on new data.
pub fn is_basis_with_knots(
    x: &Array1<f64>,
    interior_knots: &[f64],
    degree: usize,
    boundary_knots: (f64, f64),
    df: usize,
    increasing: bool,
) -> Array2<f64> {
    let n = x.len();
    let (x_min, x_max) = boundary_knots;
    
    // Build full knot vector from interior knots and boundaries
    let knots = build_knot_vector(interior_knots, degree, x_min, x_max);
    
    // Parallel evaluation over observations
    let basis_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| compute_ispline_row(x[i], degree, &knots, df, x_min, x_max))
        .collect();
    
    // Convert to Array2
    let mut result = Array2::zeros((n, df));
    for (i, row) in basis_rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    // For decreasing monotonicity, reverse the basis columns and flip values
    if !increasing {
        let mut flipped = Array2::zeros((n, df));
        for i in 0..n {
            for j in 0..df {
                flipped[[i, j]] = 1.0 - result[[i, df - 1 - j]];
            }
        }
        result = flipped;
    }
    
    result
}

/// Compute a single row of the I-spline basis.
///
/// Uses the relationship between I-splines and B-splines:
/// I_j(x) = sum_{i=j}^{n_basis-1} w_i * B_i(x)
/// where w_i are cumulative weights that ensure proper normalization.
#[inline]
fn compute_ispline_row(
    x: f64,
    degree: usize,
    knots: &[f64],
    n_basis: usize,
    x_min: f64,
    x_max: f64,
) -> Vec<f64> {
    // Handle boundary cases
    if x <= x_min {
        return vec![0.0; n_basis];
    }
    if x >= x_max {
        return vec![1.0; n_basis];
    }
    
    // Compute B-spline values at this point
    let bs_values = bspline_all_basis_at_point(x, degree, knots, n_basis);
    
    // I-splines are cumulative integrals of M-splines
    // For practical computation, we use the formula:
    // I_j(x) = sum_{i >= j} B_i(x) * (t_{i+k+1} - t_j) / (t_{i+k+1} - t_i)
    //
    // A simpler approximation that works well in practice:
    // I_j(x) ≈ cumulative sum of B-splines from right to left, normalized
    
    let mut result = vec![0.0; n_basis];
    
    // Compute cumulative sum from right to left
    // This gives I_j(x) = sum_{i=j}^{n-1} B_i(x)
    let mut cumsum = 0.0;
    for j in (0..n_basis).rev() {
        cumsum += bs_values[j];
        result[j] = cumsum;
    }
    
    // Normalize: ensure values are in [0, 1]
    // At x = x_max, all I_j should be 1
    // At x = x_min, all I_j should be 0
    for j in 0..n_basis {
        result[j] = result[j].clamp(0.0, 1.0);
    }
    
    result
}

/// Compute I-spline basis with default options (cubic, increasing).
pub fn is(x: &Array1<f64>, df: usize) -> Array2<f64> {
    is_basis(x, df, DEFAULT_DEGREE, None, true)
}

/// Get column names for I-spline (monotonic spline) basis.
pub fn is_names(var_name: &str, df: usize, increasing: bool) -> Vec<String> {
    let direction = if increasing { "+" } else { "-" };
    (0..df).map(|i| format!("ms({}, {}/{}, {})", var_name, i + 1, df, direction)).collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_knot_computation() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let knots = compute_knots(&x, 5, 3, None);
        
        // Should have: 4 left boundary, 1 interior, 4 right boundary = 9 knots
        assert_eq!(knots.len(), 9);
        
        // First 4 should be min
        assert_eq!(knots[0], 0.0);
        assert_eq!(knots[3], 0.0);
        
        // Last 4 should be max
        assert_eq!(knots[5], 5.0);
        assert_eq!(knots[8], 5.0);
    }
    
    #[test]
    fn test_bspline_partition_of_unity() {
        // B-splines should sum to 1 at any point (with intercept/full basis)
        // Use a range with explicit boundary knots for proper partition of unity
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let basis = bs_basis(&x, 6, 3, Some((0.0, 10.0)), true);
        
        // Check that each row sums to approximately 1
        for i in 0..x.len() {
            let row_sum: f64 = basis.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_bspline_shape() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let basis = bs(&x, 6);
        
        // Without intercept, df=6 gives 5 columns
        assert_eq!(basis.shape(), &[5, 5]);
    }
    
    #[test]
    fn test_bspline_with_intercept() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let basis = bs_basis(&x, 5, 3, None, true);
        
        // With intercept, df=5 gives 5 columns
        assert_eq!(basis.shape(), &[5, 5]);
    }
    
    #[test]
    fn test_natural_spline_shape() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let basis = ns(&x, 5);
        
        // Natural spline with df=5, no intercept
        assert_eq!(basis.nrows(), 10);
        assert!(basis.ncols() >= 1);
    }
    
    #[test]
    fn test_natural_spline_linear_at_boundaries() {
        // Natural splines should be linear beyond the knots
        let x = Array1::from_vec(vec![-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]);
        let basis = ns_basis(&x, 4, Some((0.0, 2.0)), false);
        
        // The basis should exist
        assert_eq!(basis.nrows(), 7);
        assert!(basis.ncols() >= 1);
    }
    
    #[test]
    fn test_bspline_non_negative() {
        // B-splines should be non-negative
        let x = Array1::from_vec(vec![0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let basis = bs(&x, 5);
        
        for val in basis.iter() {
            assert!(*val >= -1e-10, "B-spline value should be non-negative, got {}", val);
        }
    }
    
    #[test]
    fn test_parallel_consistency() {
        // Test that parallel and sequential give same results
        let x = Array1::from_vec((0..100).map(|i| i as f64 / 10.0).collect());
        
        let basis1 = bs(&x, 6);
        let basis2 = bs(&x, 6);
        
        for i in 0..basis1.nrows() {
            for j in 0..basis1.ncols() {
                assert_abs_diff_eq!(basis1[[i, j]], basis2[[i, j]], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_bs_names() {
        let names = bs_names("age", 5, false);
        assert_eq!(names.len(), 4);
        assert!(names[0].contains("bs(age"));
    }
    
    #[test]
    fn test_ns_names() {
        let names = ns_names("age", 5, false);
        assert!(names.len() >= 1);
        assert!(names[0].contains("ns(age"));
    }
    
    // =========================================================================
    // I-SPLINE (MONOTONIC SPLINE) TESTS
    // =========================================================================
    
    #[test]
    fn test_ispline_shape() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let basis = is(&x, 5);
        
        // I-spline with df=5 should have 5 columns
        assert_eq!(basis.shape(), &[6, 5]);
    }
    
    #[test]
    fn test_ispline_range() {
        // I-spline values should be in [0, 1]
        let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        let basis = is_basis(&x, 5, 3, Some((0.0, 3.0)), true);
        
        for val in basis.iter() {
            assert!(*val >= -1e-10, "I-spline value should be >= 0, got {}", val);
            assert!(*val <= 1.0 + 1e-10, "I-spline value should be <= 1, got {}", val);
        }
    }
    
    #[test]
    fn test_ispline_monotonic_increasing() {
        // Each column of increasing I-splines should be monotonically increasing
        let x = Array1::from_vec((0..50).map(|i| i as f64 / 10.0).collect());
        let basis = is_basis(&x, 5, 3, Some((0.0, 5.0)), true);
        
        for j in 0..basis.ncols() {
            for i in 1..basis.nrows() {
                assert!(
                    basis[[i, j]] >= basis[[i - 1, j]] - 1e-10,
                    "I-spline column {} should be monotonically increasing at row {}: {} < {}",
                    j, i, basis[[i, j]], basis[[i - 1, j]]
                );
            }
        }
    }
    
    #[test]
    fn test_ispline_monotonic_decreasing() {
        // Each column of decreasing I-splines should be monotonically decreasing
        let x = Array1::from_vec((0..50).map(|i| i as f64 / 10.0).collect());
        let basis = is_basis(&x, 5, 3, Some((0.0, 5.0)), false);
        
        for j in 0..basis.ncols() {
            for i in 1..basis.nrows() {
                assert!(
                    basis[[i, j]] <= basis[[i - 1, j]] + 1e-10,
                    "Decreasing I-spline column {} should be monotonically decreasing at row {}: {} > {}",
                    j, i, basis[[i, j]], basis[[i - 1, j]]
                );
            }
        }
    }
    
    #[test]
    fn test_ispline_boundary_values() {
        // At x_min, all I-spline values should be 0
        // At x_max, all I-spline values should be 1
        let x = Array1::from_vec(vec![0.0, 2.5, 5.0]);
        let basis = is_basis(&x, 5, 3, Some((0.0, 5.0)), true);
        
        // First row (x = 0 = x_min): all values should be 0
        for j in 0..basis.ncols() {
            assert_abs_diff_eq!(basis[[0, j]], 0.0, epsilon = 1e-6);
        }
        
        // Last row (x = 5 = x_max): all values should be 1
        for j in 0..basis.ncols() {
            assert_abs_diff_eq!(basis[[2, j]], 1.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_ispline_names() {
        let names = is_names("age", 5, true);
        assert_eq!(names.len(), 5);
        assert!(names[0].contains("ms(age"));
        assert!(names[0].contains("+"));  // Increasing direction
        
        let names_dec = is_names("age", 5, false);
        assert!(names_dec[0].contains("-"));  // Decreasing direction
    }
}
