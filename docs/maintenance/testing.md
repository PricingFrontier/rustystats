# Testing Strategy

This guide covers the testing approach for RustyStats, including test organization, writing effective tests, and running the test suite.

## Test Organization

```
rustystats/
├── crates/
│   └── rustystats-core/
│       └── src/
│           └── families/
│               └── poisson.rs  # Contains #[cfg(test)] mod tests
│
└── tests/
    └── python/
        ├── __init__.py
        ├── test_glm.py
        ├── test_families.py
        ├── test_links.py
        ├── test_regularization.py
        └── ...
```

### Rust Unit Tests

Located inline with the code using `#[cfg(test)]`:

```rust
// In families/poisson.rs
pub struct PoissonFamily;

impl Family for PoissonFamily { ... }

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_variance() { ... }
}
```

### Python Integration Tests

Located in `tests/python/`:

```python
# tests/python/test_families.py
import rustystats as rs

def test_poisson_variance():
    ...
```

---

## Running Tests

### Rust Tests

```bash
# All Rust tests
cargo test

# Specific crate
cargo test -p rustystats-core

# Specific test
cargo test -p rustystats-core poisson

# With output
cargo test -p rustystats-core -- --nocapture

# Release mode (faster, catches some bugs)
cargo test --release
```

### Python Tests

```bash
# All Python tests
uv run pytest tests/python/ -v

# Specific file
uv run pytest tests/python/test_glm.py -v

# Specific test
uv run pytest tests/python/test_glm.py::test_poisson_fit -v

# With coverage
uv run pytest tests/python/ --cov=rustystats

# Stop on first failure
uv run pytest tests/python/ -x
```

### Full Test Suite

```bash
# Both Rust and Python
cargo test && uv run pytest tests/python/ -v
```

---

## Writing Rust Tests

### Basic Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = array![1.0, 2.0, 3.0];
        
        // Act
        let result = process(&input);
        
        // Assert
        assert_eq!(result.len(), 3);
    }
}
```

### Testing Floating Point

Use `approx` crate for float comparisons:

```rust
use approx::assert_relative_eq;

#[test]
fn test_computation() {
    let result = compute();
    
    // Relative tolerance
    assert_relative_eq!(result, expected, epsilon = 1e-10);
    
    // Or absolute tolerance
    assert_relative_eq!(result, expected, max_relative = 1e-8);
}
```

### Testing Errors

```rust
#[test]
fn test_invalid_input_returns_error() {
    let result = validate(&invalid_input);
    
    assert!(result.is_err());
    
    // Check error type
    match result {
        Err(RustyStatsError::InvalidInput(_)) => (),
        _ => panic!("Expected InvalidInput error"),
    }
}
```

### Property-Based Testing

Test invariants:

```rust
#[test]
fn test_deviance_nonnegative() {
    let family = PoissonFamily;
    
    // Random test cases
    for _ in 0..100 {
        let y = random_positive_array(10);
        let mu = random_positive_array(10);
        
        let dev = family.deviance(&y, &mu, None);
        assert!(dev >= 0.0, "Deviance must be non-negative");
    }
}

#[test]
fn test_perfect_fit_zero_deviance() {
    let family = PoissonFamily;
    
    for _ in 0..100 {
        let y = random_positive_array(10);
        let mu = y.clone();  // Perfect fit
        
        let dev = family.deviance(&y, &mu, None);
        assert!(dev < 1e-10, "Perfect fit should have zero deviance");
    }
}
```

### Ignored/Long-Running Tests

```rust
#[test]
#[ignore]  // Skip by default
fn test_large_dataset() {
    // Long-running test
}

// Run with: cargo test -- --ignored
```

---

## Writing Python Tests

### Basic Test Structure

```python
import pytest
import numpy as np
import rustystats as rs

class TestPoissonFit:
    """Tests for Poisson GLM fitting."""
    
    def test_basic_fit(self):
        """Test basic Poisson fit converges."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
        
        result = rs.fit_glm(y, X, family="poisson")
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_predictions_positive(self):
        """Test that Poisson predictions are positive."""
        y = np.random.poisson(5, 100)
        X = np.column_stack([np.ones(100), np.random.randn(100)])
        
        result = rs.fit_glm(y, X, family="poisson")
        
        assert np.all(result.fittedvalues > 0)
```

### Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Generate sample Poisson data."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    eta = 0.5 + 0.3 * x
    y = np.random.poisson(np.exp(eta))
    X = np.column_stack([np.ones(n), x])
    return y, X

def test_with_fixture(sample_data):
    y, X = sample_data
    result = rs.fit_glm(y, X, family="poisson")
    assert result.converged
```

### Parametrized Tests

```python
@pytest.mark.parametrize("family", [
    "gaussian", "poisson", "binomial", "gamma"
])
def test_all_families_converge(family):
    """Test that all families can fit."""
    np.random.seed(42)
    y = np.abs(np.random.randn(50)) + 0.1
    if family == "binomial":
        y = (y > np.median(y)).astype(float)
    
    X = np.column_stack([np.ones(50), np.random.randn(50)])
    result = rs.fit_glm(y, X, family=family)
    
    assert result.converged
```

### Comparison with Statsmodels

```python
def test_vs_statsmodels():
    """Compare results to statsmodels."""
    import statsmodels.api as sm
    
    np.random.seed(42)
    y = np.random.poisson(5, 100).astype(float)
    X = np.column_stack([np.ones(100), np.random.randn(100, 2)])
    
    # RustyStats
    rs_result = rs.fit_glm(y, X, family="poisson")
    
    # Statsmodels
    sm_result = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    # Compare
    np.testing.assert_allclose(
        rs_result.params, 
        sm_result.params, 
        rtol=1e-5
    )
    np.testing.assert_allclose(
        rs_result.bse(), 
        sm_result.bse, 
        rtol=1e-4
    )
```

### Edge Case Tests

```python
class TestEdgeCases:
    
    def test_single_observation(self):
        """Test behavior with single observation."""
        y = np.array([1.0])
        X = np.array([[1.0]])
        
        result = rs.fit_glm(y, X, family="gaussian")
        assert result.converged
    
    def test_all_zeros_poisson(self):
        """Test Poisson with all zero response."""
        y = np.zeros(10)
        X = np.column_stack([np.ones(10), np.random.randn(10)])
        
        result = rs.fit_glm(y, X, family="poisson")
        assert result.converged
        # Predictions should still be positive
        assert np.all(result.fittedvalues > 0)
    
    def test_large_values(self):
        """Test numerical stability with large values."""
        y = np.array([1e6, 2e6, 3e6])
        X = np.column_stack([np.ones(3), [1, 2, 3]])
        
        result = rs.fit_glm(y, X, family="gaussian")
        assert result.converged
        assert not np.any(np.isnan(result.params))
```

---

## Test Categories

### 1. Unit Tests

Test individual functions in isolation:

```rust
#[test]
fn test_soft_threshold() {
    assert_eq!(soft_threshold(5.0, 2.0), 3.0);
    assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
    assert_eq!(soft_threshold(1.0, 2.0), 0.0);
}
```

### 2. Integration Tests

Test components working together:

```python
def test_formula_to_fit():
    """Test full formula API workflow."""
    data = pl.DataFrame({
        "y": [1, 2, 3, 4, 5],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["A", "B", "A", "B", "A"],
    })
    
    result = rs.glm("y ~ x + C(cat)", data, family="gaussian").fit()
    
    assert result.converged
    assert "x" in result.feature_names
```

### 3. Regression Tests

Ensure bugs don't recur:

```python
def test_issue_123_overflow():
    """Regression test for issue #123 - overflow with large exposure."""
    y = np.array([1, 2, 3])
    exposure = np.array([1e10, 1e10, 1e10])
    X = np.column_stack([np.ones(3), [1, 2, 3]])
    
    # Should not overflow
    result = rs.fit_glm(y, X, family="poisson", offset=np.log(exposure))
    assert not np.any(np.isnan(result.params))
```

### 4. Performance Tests

Verify performance characteristics:

```python
@pytest.mark.slow
def test_large_dataset_performance():
    """Test that large dataset fits in reasonable time."""
    import time
    
    n = 100000
    p = 50
    y = np.random.poisson(5, n)
    X = np.column_stack([np.ones(n), np.random.randn(n, p)])
    
    start = time.time()
    result = rs.fit_glm(y, X, family="poisson")
    elapsed = time.time() - start
    
    assert result.converged
    assert elapsed < 5.0  # Should complete in < 5 seconds
```

---

## Test Coverage

### Rust Coverage

```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Run with coverage
cargo tarpaulin -p rustystats-core --out Html
```

### Python Coverage

```bash
# Run with coverage
uv run pytest tests/python/ --cov=rustystats --cov-report=html

# View report
open htmlcov/index.html
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run Rust tests
        run: cargo test
      
      - name: Build Python package
        run: uv run maturin develop
      
      - name: Run Python tests
        run: uv run pytest tests/python/ -v
```

---

## Best Practices

1. **Test behavior, not implementation** - Focus on what the code does, not how
2. **Use descriptive test names** - `test_poisson_variance_equals_mean`
3. **One assertion per concept** - Split complex tests
4. **Test edge cases** - Empty inputs, single values, extremes
5. **Compare to reference** - Use statsmodels, R, or scipy
6. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
7. **Clean up fixtures** - Don't leave state between tests
