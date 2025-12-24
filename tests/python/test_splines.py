"""
Tests for spline basis functions in RustyStats.

Tests cover:
- B-spline basis computation
- Natural spline basis computation
- Formula integration
- Edge cases and properties
"""

import numpy as np
import pytest


class TestBSplines:
    """Tests for B-spline basis functions."""
    
    def test_bs_basic_shape(self):
        """Test that B-spline basis has correct shape."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=5)
        
        # Without intercept, df=5 gives 4 columns
        assert basis.shape == (100, 4), f"Expected (100, 4), got {basis.shape}"
    
    def test_bs_with_different_df(self):
        """Test B-splines with different degrees of freedom."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 50)
        
        for df in [3, 5, 7, 10]:
            basis = rs.bs(x, df=df)
            # Without intercept, we get df-1 columns
            expected_cols = df - 1
            assert basis.shape[1] == expected_cols, \
                f"df={df}: expected {expected_cols} cols, got {basis.shape[1]}"
    
    def test_bs_partition_of_unity(self):
        """B-splines should approximately sum to 1 at any point."""
        import rustystats as rs
        
        # Use points strictly inside the boundary for partition of unity
        x = np.linspace(1.0, 9.0, 50)  # Well inside [0, 10]
        basis = rs.bs(x, df=6, boundary_knots=(0.0, 10.0), include_intercept=True)
        
        row_sums = basis.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5,
            err_msg="B-splines should sum to 1 (partition of unity)")
    
    def test_bs_non_negative(self):
        """B-splines should be non-negative."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=6)
        
        assert np.all(basis >= -1e-10), "B-spline values should be non-negative"
    
    def test_bs_different_degrees(self):
        """Test B-splines with different polynomial degrees."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 50)
        
        for degree in [1, 2, 3]:
            basis = rs.bs(x, df=5, degree=degree)
            assert basis.shape[0] == 50
            assert basis.shape[1] >= 1
    
    def test_bs_boundary_knots(self):
        """Test B-splines with explicit boundary knots."""
        import rustystats as rs
        
        x = np.linspace(-5, 15, 100)
        basis = rs.bs(x, df=5, boundary_knots=(0.0, 10.0))
        
        assert basis.shape == (100, 4)
    
    def test_bs_include_intercept(self):
        """Test B-splines with intercept included."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 50)
        
        basis_no_int = rs.bs(x, df=5, include_intercept=False)
        basis_with_int = rs.bs(x, df=5, include_intercept=True)
        
        # With intercept should have one more column
        assert basis_with_int.shape[1] == basis_no_int.shape[1] + 1


class TestNaturalSplines:
    """Tests for natural spline basis functions."""
    
    def test_ns_basic_shape(self):
        """Test that natural spline basis has correct shape."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 100)
        basis = rs.ns(x, df=5)
        
        assert basis.shape[0] == 100
        assert basis.shape[1] >= 1
    
    def test_ns_different_df(self):
        """Test natural splines with different degrees of freedom."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 50)
        
        for df in [3, 4, 5, 6]:
            basis = rs.ns(x, df=df)
            assert basis.shape[0] == 50
            assert basis.shape[1] >= 1
    
    def test_ns_boundary_knots(self):
        """Test natural splines with explicit boundary knots."""
        import rustystats as rs
        
        x = np.linspace(-5, 15, 100)
        basis = rs.ns(x, df=5, boundary_knots=(0.0, 10.0))
        
        assert basis.shape[0] == 100


class TestSplineNames:
    """Tests for spline column name generation."""
    
    def test_bs_names(self):
        """Test B-spline name generation."""
        import rustystats as rs
        
        names = rs.bs_names("age", df=5)
        
        assert len(names) == 4  # df-1 without intercept
        assert all("bs(age" in name for name in names)
    
    def test_ns_names(self):
        """Test natural spline name generation."""
        import rustystats as rs
        
        names = rs.ns_names("income", df=4)
        
        assert len(names) >= 1
        assert all("ns(income" in name for name in names)


class TestSplineFormula:
    """Tests for spline integration with formula API."""
    
    def test_formula_with_bs(self):
        """Test B-splines in formula."""
        import rustystats as rs
        import polars as pl
        
        # Create test data
        np.random.seed(42)
        n = 100
        data = pl.DataFrame({
            "y": np.random.poisson(3, n),
            "age": np.random.uniform(20, 70, n),
            "exposure": np.ones(n),
        })
        
        # Fit model with B-spline for age
        model = rs.glm(
            "y ~ bs(age, df=5)",
            data=data,
            family="poisson",
        )
        result = model.fit()
        
        # Check that we have the right number of parameters
        # Intercept + 4 spline terms (df=5, no intercept in spline)
        assert len(result.params) >= 2
    
    def test_formula_with_ns(self):
        """Test natural splines in formula."""
        import rustystats as rs
        import polars as pl
        
        np.random.seed(42)
        n = 100
        data = pl.DataFrame({
            "y": np.random.normal(0, 1, n),
            "x": np.random.uniform(0, 10, n),
        })
        
        model = rs.glm(
            "y ~ ns(x, df=4)",
            data=data,
            family="gaussian",
        )
        result = model.fit()
        
        assert result.converged
        assert len(result.params) >= 2
    
    def test_formula_spline_with_categorical(self):
        """Test splines combined with categorical variables."""
        import rustystats as rs
        import polars as pl
        
        np.random.seed(42)
        n = 200
        data = pl.DataFrame({
            "y": np.random.poisson(2, n),
            "age": np.random.uniform(20, 70, n),
            "region": np.random.choice(["A", "B", "C"], n),
        })
        
        model = rs.glm(
            "y ~ bs(age, df=4) + C(region)",
            data=data,
            family="poisson",
        )
        result = model.fit()
        
        assert result.converged
        # Intercept + 3 spline terms + 2 region dummies
        assert len(result.params) >= 4
    
    def test_formula_multiple_splines(self):
        """Test multiple spline terms in one formula."""
        import rustystats as rs
        import polars as pl
        
        np.random.seed(42)
        n = 150
        data = pl.DataFrame({
            "y": np.random.poisson(3, n),
            "age": np.random.uniform(20, 70, n),
            "income": np.random.uniform(30000, 150000, n),
        })
        
        model = rs.glm(
            "y ~ bs(age, df=4) + ns(income, df=3)",
            data=data,
            family="poisson",
        )
        result = model.fit()
        
        assert result.converged


class TestSplineProperties:
    """Tests for mathematical properties of splines."""
    
    def test_bs_local_support(self):
        """Test that B-splines have local support (mostly zeros)."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=10)
        
        # Each row should have mostly zeros (local support)
        # For cubic splines, at most degree+1=4 basis functions are nonzero at any point
        for i in range(len(x)):
            nonzero = np.sum(basis[i, :] > 1e-10)
            assert nonzero <= 4, f"Row {i} has {nonzero} nonzero values, expected <= 4"
    
    def test_bs_smoothness(self):
        """Test that adjacent basis values don't jump too much (smoothness)."""
        import rustystats as rs
        
        # Use fine grid and explicit boundaries for smooth evaluation
        x = np.linspace(0.5, 9.5, 1000)  # Stay away from boundaries
        basis = rs.bs(x, df=6, boundary_knots=(0.0, 10.0))
        
        # Check that differences between adjacent rows are small
        diffs = np.abs(np.diff(basis, axis=0))
        max_diff = np.max(diffs)
        
        # Max diff should be reasonably small for smooth splines
        # With 1000 points over range 9, step is 0.009, so max change should be moderate
        assert max_diff < 0.2, f"Max difference {max_diff} too large for smooth splines"
    
    def test_spline_reproducibility(self):
        """Test that spline computation is reproducible."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 100)
        
        basis1 = rs.bs(x, df=5)
        basis2 = rs.bs(x, df=5)
        
        np.testing.assert_array_equal(basis1, basis2)


class TestSplineEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_array(self):
        """Test splines with small arrays."""
        import rustystats as rs
        
        x = np.array([1.0, 2.0, 3.0])
        basis = rs.bs(x, df=3)
        
        assert basis.shape[0] == 3
    
    def test_single_value(self):
        """Test splines with single value (edge case)."""
        import rustystats as rs
        
        x = np.array([5.0])
        basis = rs.bs(x, df=3)
        
        assert basis.shape[0] == 1
    
    def test_identical_values(self):
        """Test splines when all x values are identical."""
        import rustystats as rs
        
        x = np.array([5.0, 5.0, 5.0, 5.0])
        basis = rs.bs(x, df=3, boundary_knots=(0.0, 10.0))
        
        assert basis.shape[0] == 4
        # All rows should be identical
        for i in range(1, 4):
            np.testing.assert_array_almost_equal(basis[0, :], basis[i, :])
    
    def test_extrapolation(self):
        """Test splines with values outside boundary knots."""
        import rustystats as rs
        
        # Values outside [0, 10]
        x = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        basis = rs.bs(x, df=4, boundary_knots=(0.0, 10.0))
        
        assert basis.shape[0] == 5


class TestSplineTerm:
    """Tests for the SplineTerm class."""
    
    def test_spline_term_bs(self):
        """Test SplineTerm with B-splines."""
        import rustystats as rs
        
        term = rs.SplineTerm("age", spline_type="bs", df=5)
        
        x = np.linspace(20, 70, 50)
        basis, names = term.transform(x)
        
        assert basis.shape[0] == 50
        assert len(names) == basis.shape[1]
    
    def test_spline_term_ns(self):
        """Test SplineTerm with natural splines."""
        import rustystats as rs
        
        term = rs.SplineTerm("income", spline_type="ns", df=4)
        
        x = np.linspace(30000, 150000, 100)
        basis, names = term.transform(x)
        
        assert basis.shape[0] == 100
        assert len(names) == basis.shape[1]
    
    def test_spline_term_repr(self):
        """Test SplineTerm string representation."""
        import rustystats as rs
        
        term_bs = rs.SplineTerm("age", spline_type="bs", df=5)
        term_ns = rs.SplineTerm("age", spline_type="ns", df=4)
        
        assert "bs(age" in repr(term_bs)
        assert "ns(age" in repr(term_ns)


class TestSplinePerformance:
    """Performance-related tests for splines."""
    
    def test_large_array(self):
        """Test splines with large array (performance check)."""
        import rustystats as rs
        import time
        
        x = np.random.uniform(0, 10, 100000)
        
        start = time.time()
        basis = rs.bs(x, df=10)
        elapsed = time.time() - start
        
        assert basis.shape == (100000, 9)
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Spline computation took {elapsed:.2f}s, expected < 1s"
    
    def test_many_basis_functions(self):
        """Test splines with many basis functions."""
        import rustystats as rs
        
        x = np.linspace(0, 10, 1000)
        basis = rs.bs(x, df=20)
        
        assert basis.shape == (1000, 19)
