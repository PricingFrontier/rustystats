"""High-assurance tests for penalized-spline helper mathematics."""

from __future__ import annotations

import numpy as np
import pytest
import rustystats as rs
import rustystats.splines as splines_mod
from rustystats.exceptions import ValidationError
from rustystats.splines import SplineTerm


class TestDifferenceMatrix:
    def test_order_zero_is_identity(self):
        np.testing.assert_array_equal(rs.difference_matrix(4, order=0), np.eye(4))

    def test_order_one_matches_hand_calculation(self):
        expected = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(rs.difference_matrix(4, order=1), expected)

    def test_order_two_matches_hand_calculation(self):
        expected = np.array(
            [
                [1.0, -2.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, -2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, -2.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(rs.difference_matrix(5, order=2), expected)

    def test_order_three_matches_repeated_difference_reference(self):
        expected = np.array(
            [
                [-1.0, 3.0, -3.0, 1.0, 0.0],
                [0.0, -1.0, 3.0, -3.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(rs.difference_matrix(5, order=3), expected)

    def test_too_few_coefficients_returns_empty_difference_operator(self):
        D = rs.difference_matrix(2, order=2)
        assert D.shape == (0, 2)


class TestPenaltyMatrix:
    def test_penalty_matrix_is_d_transpose_d(self):
        D = rs.difference_matrix(5, order=2)
        np.testing.assert_array_equal(rs.penalty_matrix(5, order=2), D.T @ D)

    def test_penalty_matrix_is_symmetric_positive_semidefinite(self):
        S = rs.penalty_matrix(7, order=2)
        np.testing.assert_allclose(S, S.T, atol=0.0)
        assert np.min(np.linalg.eigvalsh(S)) >= -1e-12

    def test_second_difference_penalty_null_space_contains_constant_and_linear_terms(self):
        S = rs.penalty_matrix(6, order=2)
        constant = np.ones(6)
        linear = np.arange(6, dtype=np.float64)
        assert constant @ S @ constant == pytest.approx(0.0, abs=1e-12)
        assert linear @ S @ linear == pytest.approx(0.0, abs=1e-12)

    def test_first_difference_penalty_has_constant_null_space_only(self):
        S = rs.penalty_matrix(5, order=1)
        constant = np.ones(5)
        linear = np.arange(5, dtype=np.float64)
        assert constant @ S @ constant == pytest.approx(0.0, abs=1e-12)
        assert linear @ S @ linear > 0.0


class TestGcvScore:
    def test_gcv_score_matches_formula(self):
        assert rs.gcv_score(deviance=100.0, n=1000, edf=10.0) == pytest.approx(
            1000.0 * 100.0 / 990.0**2
        )

    def test_gcv_score_floors_degenerate_denominator(self):
        assert rs.gcv_score(deviance=7.0, n=5, edf=6.0) == pytest.approx(35.0)
        assert rs.gcv_score(deviance=7.0, n=5, edf=5.0) == pytest.approx(35.0)


class TestComputeEdf:
    def test_nonpositive_lambda_returns_full_basis_dimension(self):
        xtwx = np.diag([2.0, 3.0, 4.0])
        penalty = rs.penalty_matrix(3, order=1)
        assert rs.compute_edf(xtwx, penalty, lambda_=0.0) == pytest.approx(3.0)
        assert rs.compute_edf(xtwx, penalty, lambda_=-1.0) == pytest.approx(3.0)

    def test_zero_penalty_preserves_full_edf_for_positive_lambda(self):
        xtwx = np.diag([2.0, 3.0, 4.0])
        penalty = np.zeros((3, 3))
        assert rs.compute_edf(xtwx, penalty, lambda_=100.0) == pytest.approx(3.0)

    def test_edf_decreases_as_lambda_increases_for_first_difference_penalty(self):
        xtwx = np.eye(5)
        penalty = rs.penalty_matrix(5, order=1)
        low = rs.compute_edf(xtwx, penalty, lambda_=0.1)
        high = rs.compute_edf(xtwx, penalty, lambda_=100.0)
        assert 1.0 < high < low < 5.0

    def test_edf_matches_trace_formula_for_diagonal_case(self):
        xtwx = np.diag([2.0, 4.0])
        penalty = np.diag([1.0, 3.0])
        lambda_ = 0.5
        expected = np.trace(np.linalg.inv(xtwx + lambda_ * penalty) @ xtwx)
        assert rs.compute_edf(xtwx, penalty, lambda_) == pytest.approx(expected)

    def test_singular_penalized_system_warns_and_returns_nan(self):
        xtwx = np.zeros((2, 2))
        penalty = np.zeros((2, 2))
        with pytest.warns(RuntimeWarning, match="EDF computation failed"):
            edf = rs.compute_edf(xtwx, penalty, lambda_=1.0)
        assert np.isnan(edf)


class TestSplineBasisContracts:
    def test_bs_and_ns_reject_conflicting_df_and_k(self):
        x = np.linspace(0.0, 1.0, 6)

        with pytest.raises(ValidationError, match=r"either 'df'.*'k'"):
            rs.bs(x, df=4, k=6)

        with pytest.raises(ValidationError, match=r"either 'df'.*'k'"):
            rs.ns(x, df=4, k=6)

    def test_bs_rejects_invalid_monotonicity_and_accepts_valid_constraints(self):
        x = np.linspace(0.0, 1.0, 8)

        with pytest.raises(ValidationError, match="monotonicity"):
            rs.bs(x, df=4, monotonicity="flat")

        increasing = rs.bs(x, df=4, monotonicity="increasing")
        decreasing = rs.bs(x, df=4, monotonicity="decreasing")
        assert increasing.shape == decreasing.shape
        assert increasing.shape[0] == len(x)

    def test_explicit_bs_knots_drop_intercept_column_by_default(self):
        x = np.linspace(0.0, 1.0, 9)

        with_intercept = rs.bs(
            x,
            knots=[0.25, 0.5, 0.75],
            degree=3,
            boundary_knots=(0.0, 1.0),
            include_intercept=True,
        )
        without_intercept = rs.bs(
            x,
            knots=[0.25, 0.5, 0.75],
            degree=3,
            boundary_knots=(0.0, 1.0),
            include_intercept=False,
        )

        assert without_intercept.shape[0] == with_intercept.shape[0] == len(x)
        assert without_intercept.shape[1] == with_intercept.shape[1] - 1
        np.testing.assert_allclose(without_intercept, with_intercept[:, 1:])

    def test_spline_term_reuses_explicit_knots_and_reports_metadata(self):
        term = SplineTerm("age", spline_type="bs", df=5, degree=3, boundary_knots=(0.0, 10.0))
        term._computed_internal_knots = [3.0, 6.0]

        train_basis, train_names = term.transform(np.linspace(0.0, 10.0, 7))
        score_basis, score_names = term.transform(np.array([-1.0, 2.0, 11.0]))
        info = term.get_knot_info()

        assert train_basis.shape[1] == len(train_names)
        assert score_basis.shape[1] == len(score_names)
        assert info["boundary_knots"] == [0.0, 10.0]
        assert info["knots"] == [3.0, 6.0]
        assert info["degree"] == 3

    def test_spline_term_smooth_metadata_and_name_fallback(self, monkeypatch):
        term = SplineTerm("age", spline_type="bs", df=4, degree=3, monotonicity="decreasing")
        term._is_smooth = True
        term._lambda = 0.25
        term._edf = 2.5

        basis, names = term.transform(np.linspace(0.0, 1.0, 8))
        info = term.get_knot_info()

        assert basis.shape[1] == len(names)
        assert all("k, -" in name for name in names)
        assert info["is_smooth"] is True
        assert info["lambda"] == pytest.approx(0.25)
        assert info["edf"] == pytest.approx(2.5)
        assert repr(term) == "bs(age, df=4, monotonicity='decreasing')"

        monkeypatch.setattr(splines_mod, "bs_names", lambda *_args, **_kwargs: ["too-short"])
        fallback = SplineTerm("veh_age", spline_type="bs", df=4)
        fallback_basis, fallback_names = fallback.transform(np.linspace(0.0, 1.0, 8))

        assert len(fallback_names) == fallback_basis.shape[1]
        assert all(name.startswith("bs(veh_age, ") for name in fallback_names)

    def test_natural_spline_monotonicity_rejects_and_ms_defaults_to_increasing(self):
        x = np.linspace(0.0, 1.0, 8)
        ns_term = SplineTerm("age", spline_type="ns", df=4, monotonicity="increasing")

        with pytest.raises(ValidationError, match="not supported"):
            ns_term.transform(x)

        ms_term = SplineTerm("age", spline_type="ms", df=4)
        basis, names = ms_term.transform(x)
        info = ms_term.get_knot_info()

        assert basis.shape[1] == len(names)
        assert all(name.startswith("ms(age, ") for name in names)
        assert info["type"] == "ms"
        assert repr(ms_term) == "ms(age, df=4, monotonicity='increasing')"

    def test_spline_term_rejects_unknown_type(self):
        with pytest.raises(ValidationError, match="spline_type"):
            SplineTerm("age", spline_type="cr")

    def test_spline_term_ms_smooth_and_fallback_branches(self):
        x = np.linspace(0.0, 1.0, 8)
        smooth_ms = SplineTerm("age", spline_type="ms", df=4)
        smooth_ms._is_smooth = True
        basis, names = smooth_ms.transform(x)

        assert basis.shape[1] == len(names)
        assert all(name.startswith("ms(age, ") for name in names)

        fallback_ms = SplineTerm("age", spline_type="ms", df=4, boundary_knots=(0.0, 1.0))
        fallback_ms.transform(x)
        fallback_ms._computed_internal_knots = None
        fallback_basis, fallback_names = fallback_ms.transform(x)

        assert fallback_basis.shape[1] == len(fallback_names)

    def test_spline_term_repr_variants_and_mutated_unknown_type(self):
        assert repr(SplineTerm("age", spline_type="bs", df=4, degree=2)) == (
            "bs(age, df=4, degree=2)"
        )
        assert repr(SplineTerm("age", spline_type="ns", df=4)) == "ns(age, df=4)"

        term = SplineTerm("age", spline_type="bs", df=4)
        term.spline_type = "cr"
        with pytest.raises(ValidationError, match="Unknown spline_type"):
            term.transform(np.linspace(0.0, 1.0, 8))
