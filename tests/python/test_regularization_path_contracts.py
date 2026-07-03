"""Contract tests for regularization-path selection utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rustystats._rustystats as rust_backend
import rustystats.regularization_path as reg_path
from rustystats.exceptions import FittingError, ValidationError
from rustystats.regularization_path import (
    ALPHA_MAX_FLOOR,
    RegularizationPathResult,
    _alpha_family_base,
    _fit_null_intercept,
    _glm_score_gradient_factor,
    compute_alpha_max,
    compute_deviance,
    compute_standardization,
    compute_standardization_with_ridge_diag,
    create_cv_folds,
    fit_cv_regularization_path,
    fit_cv_te_regularization_path,
    generate_alpha_path,
    penalized_column_mask,
    reset_fold_local_spline_state,
    select_optimal_alpha,
    solver_standardization,
)


def _force_python_standardization(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def block_rust_standardization(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rustystats._rustystats" and "compute_standardization_py" in fromlist:
            raise ImportError("force Python fallback")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_rust_standardization)


def _path_result(alpha: float, mean: float, se: float = 0.0) -> RegularizationPathResult:
    return RegularizationPathResult(
        alpha=alpha,
        l1_ratio=1.0,
        cv_deviance_mean=mean,
        cv_deviance_se=se,
        n_nonzero=0,
        max_coef=0.0,
    )


class TestRegularizationPathUtilities:
    def test_penalized_column_mask_excludes_only_intercept(self):
        np.testing.assert_array_equal(
            penalized_column_mask(4, fit_intercept=True),
            np.array([False, True, True, True]),
        )
        np.testing.assert_array_equal(
            penalized_column_mask(4, fit_intercept=False),
            np.array([True, True, True, True]),
        )
        np.testing.assert_array_equal(penalized_column_mask(0, fit_intercept=True), np.array([]))

    def test_generate_alpha_path_is_descending_geometric_grid(self):
        path = generate_alpha_path(alpha_max=100.0, n_alphas=5, alpha_min_ratio=0.01)

        assert path[0] == pytest.approx(100.0)
        assert path[-1] == pytest.approx(1.0)
        assert np.all(np.diff(path) < 0.0)
        ratios = path[1:] / path[:-1]
        np.testing.assert_allclose(ratios, np.repeat(ratios[0], 4), rtol=1e-14)

    def test_generate_alpha_path_handles_single_point_and_floor(self):
        path = generate_alpha_path(alpha_max=ALPHA_MAX_FLOOR, n_alphas=1, alpha_min_ratio=0.1)

        np.testing.assert_allclose(path, np.array([ALPHA_MAX_FLOOR]))

    def test_create_cv_folds_is_deterministic_balanced_and_exhaustive(self):
        folds = create_cv_folds(11, 4, seed=123)
        folds_again = create_cv_folds(11, 4, seed=123)

        val_sets = [tuple(val.tolist()) for _train, val in folds]
        assert val_sets == [tuple(val.tolist()) for _train, val in folds_again]
        assert sorted(len(val) for _train, val in folds) == [2, 3, 3, 3]

        all_val = np.concatenate([val for _train, val in folds])
        np.testing.assert_array_equal(np.sort(all_val), np.arange(11))
        for train, val in folds:
            assert set(train).isdisjoint(set(val))
            np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(11))

    @pytest.mark.parametrize(
        ("family", "expected"),
        [
            ("normal", "gaussian"),
            ("NegativeBinomial(theta=2.5)", "negbinomial"),
            ("negative_binomial", "negbinomial"),
            ("NB", "negbinomial"),
            ("Poisson", "poisson"),
        ],
    )
    def test_alpha_family_base_normalizes_public_aliases(self, family, expected):
        assert _alpha_family_base(family) == expected

    def test_solver_standardization_zeros_center_when_intercept_absorbs_it(self):
        center = np.array([0.0, 10.0, -3.0])
        scale = np.array([1.0, 2.0, 4.0])

        solver_center, solver_scale = solver_standardization(center, scale, fit_intercept=True)
        np.testing.assert_array_equal(solver_center, np.zeros_like(center))
        np.testing.assert_array_equal(solver_scale, scale)

        solver_center, solver_scale = solver_standardization(center, scale, fit_intercept=False)
        np.testing.assert_array_equal(solver_center, center)
        np.testing.assert_array_equal(solver_scale, scale)

    def test_standardization_rejects_invalid_shapes_and_weights(self):
        with pytest.raises(ValidationError, match="2D"):
            compute_standardization_with_ridge_diag(np.ones(3))

        with pytest.raises(ValidationError, match="weights"):
            compute_standardization_with_ridge_diag(np.ones((3, 2)), weights=np.ones(2))

        with pytest.raises(ValidationError, match="positive finite"):
            compute_standardization_with_ridge_diag(np.ones((3, 2)), weights=np.zeros(3))

        with pytest.raises(ValidationError, match="pen_mask"):
            compute_standardization_with_ridge_diag(
                np.ones((3, 2)),
                pen_mask=np.array([True, False, True]),
            )

    def test_standardization_handles_empty_and_all_unpenalized_designs(self):
        center, scale, ridge_diag = compute_standardization_with_ridge_diag(np.ones((3, 0)))

        assert center.size == scale.size == ridge_diag.size == 0

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(
            np.ones((3, 1)),
            pen_mask=np.array([False]),
            fit_intercept=False,
        )

        np.testing.assert_array_equal(center, np.array([0.0]))
        np.testing.assert_array_equal(scale, np.array([1.0]))
        np.testing.assert_array_equal(ridge_diag, np.array([0.0]))

    def test_standardization_leaves_constant_penalized_columns_unscaled(self):
        x = np.array(
            [
                [1.0, 10.0, 0.0],
                [1.0, 10.0, 2.0],
                [1.0, 10.0, 4.0],
            ]
        )

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(x)

        assert center[0] == 0.0
        assert scale[0] == 1.0
        assert center[1] == 0.0
        assert scale[1] == 1.0
        assert center[2] == pytest.approx(2.0)
        assert scale[2] == pytest.approx(np.sqrt(8.0 / 3.0))
        assert ridge_diag[1] == pytest.approx(300.0)
        assert ridge_diag[2] == pytest.approx(3.0)

    def test_standardization_python_fallback_matches_weighted_oracle(self, monkeypatch):
        _force_python_standardization(monkeypatch)
        x = np.array(
            [
                [1.0, 10.0, 5.0],
                [1.0, 10.0, 7.0],
                [1.0, 10.0, 9.0],
            ]
        )
        weights = np.array([1.0, 2.0, 1.0])

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(
            x,
            weights=weights,
            pen_mask=np.array([False, True, True]),
            fit_intercept=False,
        )

        np.testing.assert_array_equal(center, np.zeros(3))
        assert scale[1] == 1.0
        assert scale[2] == pytest.approx(np.sqrt(2.0))
        assert ridge_diag[1] == pytest.approx(400.0)
        assert ridge_diag[2] == pytest.approx(102.0)

        center_only, scale_only = compute_standardization(
            x,
            weights=weights,
            pen_mask=np.array([False, True, True]),
            fit_intercept=False,
        )
        np.testing.assert_allclose(center_only, center)
        np.testing.assert_allclose(scale_only, scale)

    def test_standardization_python_fallback_covers_active_column_layouts(self, monkeypatch):
        _force_python_standardization(monkeypatch)

        x = np.array(
            [
                [1.0, 2.0, 10.0],
                [3.0, 4.0, 20.0],
                [5.0, 6.0, 30.0],
            ]
        )

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(
            x[:, :2],
            pen_mask=np.array([True, True]),
            fit_intercept=False,
        )
        np.testing.assert_array_equal(center, np.zeros(2))
        assert np.all(scale > 0.0)
        assert np.all(ridge_diag > 0.0)

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(
            x,
            pen_mask=np.array([True, False, True]),
            fit_intercept=False,
        )
        assert center[1] == 0.0
        assert scale[1] == 1.0
        assert ridge_diag[1] == 0.0
        assert scale[0] > 0.0
        assert scale[2] > 0.0

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(x)
        np.testing.assert_allclose(center[1:], [4.0, 20.0])
        np.testing.assert_allclose(ridge_diag[1:], [3.0, 3.0])
        assert np.all(scale[1:] > 0.0)

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(
            np.ones((3, 2)),
            pen_mask=np.array([False, False]),
            fit_intercept=False,
        )
        np.testing.assert_array_equal(center, np.zeros(2))
        np.testing.assert_array_equal(scale, np.ones(2))
        np.testing.assert_array_equal(ridge_diag, np.zeros(2))

        center, scale, ridge_diag = compute_standardization_with_ridge_diag(np.ones((3, 2)))
        np.testing.assert_array_equal(center, np.zeros(2))
        np.testing.assert_array_equal(scale, np.ones(2))
        np.testing.assert_array_equal(ridge_diag, np.array([0.0, 3.0]))


class TestRegularizationPathMath:
    def test_fit_null_intercept_poisson_log_closed_form_respects_offset_and_weights(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        offset = np.log(np.array([0.5, 1.0, 2.0, 4.0]))
        weights = np.array([1.0, 2.0, 0.5, 1.5])

        beta = _fit_null_intercept(
            y,
            "poisson",
            "log",
            offset,
            weights,
            var_power=1.5,
            theta=1.0,
        )

        expected = np.log(np.sum(weights * y) / np.sum(weights * np.exp(offset)))
        assert beta == pytest.approx(expected)

    def test_fit_null_intercept_poisson_log_returns_floor_for_no_weighted_claims(self):
        beta = _fit_null_intercept(
            np.zeros(3),
            "poisson",
            "log",
            np.zeros(3),
            np.ones(3),
            var_power=1.5,
            theta=1.0,
        )
        assert np.isfinite(beta)
        assert np.exp(beta) < ALPHA_MAX_FLOOR

    def test_fit_null_intercept_gaussian_identity_closed_form_and_weight_validation(self):
        y = np.array([10.0, 12.0, 20.0])
        offset = np.array([1.0, 2.0, 4.0])
        weights = np.array([1.0, 2.0, 1.0])

        beta = _fit_null_intercept(
            y,
            "gaussian",
            "identity",
            offset,
            weights,
            var_power=1.5,
            theta=1.0,
        )

        assert beta == pytest.approx(np.sum(weights * (y - offset)) / np.sum(weights))

        with pytest.raises(ValidationError, match="sum\\(weights\\)"):
            _fit_null_intercept(
                y,
                "gaussian",
                "identity",
                offset,
                np.zeros_like(weights),
                var_power=1.5,
                theta=1.0,
            )

    def test_score_gradient_factor_matches_canonical_and_noncanonical_oracles(self):
        y = np.array([1.0, 3.0, 5.0])
        mu = np.array([1.5, 2.0, 4.0])
        weights = np.array([2.0, 0.5, 1.0])

        gaussian = _glm_score_gradient_factor(
            y,
            mu,
            "gaussian",
            "identity",
            var_power=1.5,
            theta=2.0,
            weights=weights,
        )
        np.testing.assert_allclose(gaussian, weights * (y - mu))

        gamma_log = _glm_score_gradient_factor(
            y,
            mu,
            "gamma",
            "log",
            var_power=1.5,
            theta=2.0,
            weights=weights,
        )
        np.testing.assert_allclose(gamma_log, weights * (y - mu) / mu)

        binomial_logit = _glm_score_gradient_factor(
            np.array([0.0, 1.0]),
            np.array([0.25, 0.75]),
            "binomial",
            "logit",
            var_power=1.5,
            theta=2.0,
            weights=np.array([1.0, 3.0]),
        )
        np.testing.assert_allclose(binomial_logit, np.array([-0.25, 0.75]))

        tweedie_log = _glm_score_gradient_factor(
            y,
            mu,
            "tweedie",
            "log",
            var_power=1.5,
            theta=2.0,
            weights=weights,
        )
        np.testing.assert_allclose(tweedie_log, weights * (y - mu) / np.sqrt(mu))

        negbinomial_log = _glm_score_gradient_factor(
            y,
            mu,
            "negbinomial",
            "log",
            var_power=1.5,
            theta=2.0,
            weights=weights,
        )
        np.testing.assert_allclose(
            negbinomial_log,
            weights * (y - mu) * mu / (mu + mu * mu / 2.0),
        )

    def test_score_gradient_factor_rejects_unsupported_link_and_family(self):
        y = np.array([1.0])
        mu = np.array([1.0])
        weights = np.array([1.0])

        with pytest.raises(ValidationError, match="unsupported link"):
            _glm_score_gradient_factor(y, mu, "gaussian", "sqrt", 1.5, 1.0, weights)

        with pytest.raises(ValidationError, match="unsupported family"):
            _glm_score_gradient_factor(y, mu, "custom", "identity", 1.5, 1.0, weights)

    def test_compute_deviance_matches_gaussian_and_poisson_formulas(self):
        y = np.array([0.0, 1.0, 4.0])
        mu = np.array([0.5, 1.5, 3.0])

        gaussian = compute_deviance(y, mu, "gaussian")
        assert gaussian == pytest.approx(float(np.mean((y - mu) ** 2)))

        poisson_terms = np.array(
            [
                2.0 * (mu[0] - y[0]),
                2.0 * (y[1] * np.log(y[1] / mu[1]) - (y[1] - mu[1])),
                2.0 * (y[2] * np.log(y[2] / mu[2]) - (y[2] - mu[2])),
            ]
        )
        weights = np.array([1.0, 2.0, 3.0])
        poisson = compute_deviance(y, mu, "poisson", weights=weights)
        assert poisson == pytest.approx(float(np.sum(weights * poisson_terms) / np.sum(weights)))

    @pytest.mark.parametrize(
        "family",
        [
            "normal",
            "quasipoisson",
            "gamma",
            "binomial",
            "quasibinomial",
            "negbinomial",
            "negative_binomial",
            "tweedie",
        ],
    )
    def test_compute_deviance_dispatches_supported_family_aliases(self, family):
        if "binomial" in family and "negative" not in family and "neg" not in family:
            y = np.array([0.0, 1.0, 1.0])
            mu = np.array([0.2, 0.7, 0.8])
        else:
            y = np.array([1.0, 2.0, 4.0])
            mu = np.array([1.2, 2.5, 3.5])

        deviance = compute_deviance(
            y,
            mu,
            family,
            theta=2.0,
            var_power=1.5,
            weights=np.array([1.0, 2.0, 1.0]),
            allow_extended_tweedie=True,
        )

        assert np.isfinite(deviance)
        assert deviance >= 0.0

    def test_compute_alpha_max_lasso_and_ridge_oracles_and_validation(self):
        x = np.array(
            [
                [1.0, 0.0],
                [1.0, 2.0],
                [1.0, 4.0],
            ]
        )
        y = np.array([0.0, 1.0, 3.0])
        weights = np.array([1.0, 2.0, 1.0])
        center = np.array([0.0, 2.0])
        scale = np.array([1.0, np.sqrt(2.0)])

        alpha = compute_alpha_max(
            x,
            y,
            0.5,
            family="gaussian",
            link="",
            weights=weights,
            intercept_col=0,
            center=center,
            scale=scale,
        )
        expected_score = abs((x[:, 1] @ (weights * (y - 1.25))) / np.sqrt(2.0))
        assert alpha == pytest.approx(expected_score / 0.5)

        ridge = compute_alpha_max(
            x[:, 1:],
            y,
            0.0,
            family="gaussian",
            link="identity",
            weights=weights,
            intercept_col=None,
            ridge_xtx_diag=np.array([12.0]),
        )
        assert ridge == pytest.approx(120.0)

        assert (
            compute_alpha_max(
                x[:, :1],
                y,
                1.0,
                family="gaussian",
                link="identity",
                weights=weights,
                intercept_col=0,
            )
            == ALPHA_MAX_FLOOR
        )

        with pytest.raises(ValidationError, match="pen_mask"):
            compute_alpha_max(
                x,
                y,
                1.0,
                family="gaussian",
                link="identity",
                pen_mask=np.array([True]),
            )

        with pytest.raises(ValidationError, match="center and scale"):
            compute_alpha_max(
                x,
                y,
                1.0,
                family="gaussian",
                link="identity",
                center=center,
            )

        with pytest.raises(ValidationError, match="center/scale"):
            compute_alpha_max(
                x,
                y,
                1.0,
                family="gaussian",
                link="identity",
                center=np.zeros(1),
                scale=np.ones(1),
            )

        with pytest.raises(ValidationError, match="scale > 0"):
            compute_alpha_max(
                x,
                y,
                1.0,
                family="gaussian",
                link="identity",
                center=center,
                scale=np.array([1.0, 0.0]),
            )

        with pytest.raises(ValidationError, match="ridge_xtx_diag"):
            compute_alpha_max(
                x,
                y,
                0.0,
                family="gaussian",
                link="identity",
                ridge_xtx_diag=np.array([1.0]),
            )

        with pytest.raises(ValidationError, match="non-negative"):
            compute_alpha_max(
                x,
                y,
                0.0,
                family="gaussian",
                link="identity",
                ridge_xtx_diag=np.array([1.0, -1.0]),
            )

        assert (
            compute_alpha_max(
                x[:, :1],
                y,
                0.0,
                family="gaussian",
                link="identity",
                weights=weights,
                intercept_col=0,
            )
            == ALPHA_MAX_FLOOR
        )

    def test_compute_deviance_rejects_unknown_family_and_bad_weight_total(self):
        y = np.array([1.0, 2.0])
        mu = np.array([1.0, 2.0])

        with pytest.raises(ValidationError, match="Unknown family"):
            compute_deviance(y, mu, "custom")

        assert np.isinf(compute_deviance(y, mu, "gaussian", weights=np.zeros(2)))


class TestRegularizationPathSelection:
    def test_select_min_ignores_failed_path_points(self):
        selected = select_optimal_alpha(
            [
                _path_result(100.0, np.inf),
                _path_result(10.0, 1.2),
                _path_result(1.0, 0.8),
            ],
            selection="min",
        )

        assert selected.alpha == 1.0

    def test_select_one_standard_error_returns_largest_alpha_within_threshold(self):
        selected = select_optimal_alpha(
            [
                _path_result(100.0, 0.995),
                _path_result(10.0, 1.00, se=0.15),
                _path_result(1.0, 0.98, se=0.02),
                _path_result(0.1, 1.05),
            ],
            selection="1se",
        )

        assert selected.alpha == 100.0

    def test_select_alpha_rejects_empty_or_unknown_selection(self):
        with pytest.raises(FittingError, match="All regularization path fits failed"):
            select_optimal_alpha([_path_result(1.0, np.inf)])

        with pytest.raises(ValidationError, match="Unknown selection"):
            select_optimal_alpha([_path_result(1.0, 1.0)], selection="median")

    def test_reset_fold_local_spline_state_clears_mutable_fit_artifacts(self):
        spline = SimpleNamespace(
            _computed_boundary_knots=[0.0, 1.0],
            _computed_internal_knots=[0.5],
            _penalty_matrix=np.eye(2),
            _lambda=0.1,
            _edf=1.5,
        )
        parsed = SimpleNamespace(
            spline_terms=[spline],
            _spline_by_var={"age": spline},
            _te_by_var={"brand": object()},
        )

        reset_fold_local_spline_state(parsed)

        assert spline._computed_boundary_knots is None
        assert spline._computed_internal_knots is None
        assert spline._penalty_matrix is None
        assert spline._lambda is None
        assert spline._edf is None
        assert parsed._spline_by_var is None
        assert parsed._te_by_var is None

    def test_fit_cv_regularization_path_forwards_contract_to_rust_boundary(
        self, monkeypatch, capsys
    ):
        calls = {}

        def fake_fit_cv_path(*args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            alphas = list(args[8])
            return {
                "alphas": alphas,
                "cv_deviance_mean": [float(i) for i in range(len(alphas), 0, -1)],
                "cv_deviance_se": [0.01] * len(alphas),
                "cv_fold_scores": [[float(i), float(i + 1)] for i, _alpha in enumerate(alphas)],
                "profile": {"mocked": True},
            }

        monkeypatch.setattr(rust_backend, "fit_cv_path_py", fake_fit_cv_path)
        glm = SimpleNamespace(
            X=np.array(
                [
                    [1.0, 0.0, 3.0],
                    [1.0, 1.0, 2.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 3.0, 0.0],
                ]
            ),
            y=np.array([0.0, 1.0, 2.0, 3.0]),
            family="gaussian",
            link="identity",
            var_power=1.5,
            theta=None,
            offset=None,
            weights=np.array([1.0, 2.0, 1.0, 2.0]),
            allow_extended_tweedie=True,
            intercept=True,
            feature_names=["Intercept", "pos(x)", "neg(z)"],
        )

        info = fit_cv_regularization_path(
            glm,
            cv=2,
            selection="min",
            regularization="elastic_net",
            n_alphas=3,
            alpha_min_ratio=0.1,
            l1_ratio=None,
            max_iter=7,
            tol=1e-5,
            seed=None,
            include_unregularized=True,
            verbose=True,
            standardize=True,
        )

        assert info.selected_alpha == 0.0
        assert info.regularization_type == "none"
        assert info.selected_l1_ratio == pytest.approx(0.5)
        assert info.n_folds == 2
        assert info.cv_max_iter == 7
        assert info.cv_tol == pytest.approx(1e-5)
        assert info.cv_fold_scores is not None
        assert info.cv_profile == {"mocked": True}
        assert calls["args"][9] == pytest.approx(0.5)
        assert calls["args"][10] == 2
        assert calls["args"][11] == 7
        assert calls["args"][12] == pytest.approx(1e-5)
        assert calls["args"][13] == 42
        assert calls["kwargs"]["nonneg_indices"] == [1]
        assert calls["kwargs"]["nonpos_indices"] == [2]
        assert calls["kwargs"]["allow_extended_tweedie"] is True
        assert calls["kwargs"]["fit_intercept"] is True
        np.testing.assert_array_equal(calls["kwargs"]["center"], np.zeros(3))
        assert calls["kwargs"]["scale"] is not None
        assert "Using Rust parallel CV" in capsys.readouterr().out

        with pytest.raises(ValidationError, match="Unknown regularization"):
            fit_cv_regularization_path(glm, regularization="adaptive")

    def _mock_te_designs(self, monkeypatch):
        calls = []

        def fake_build_fold_design_matrices(
            data,
            parsed,
            train_idx,
            val_idx,
            raw_exposure=None,
            seed=None,
        ):
            calls.append(
                {
                    "data": data,
                    "parsed": parsed,
                    "train_idx": train_idx.copy(),
                    "val_idx": val_idx.copy(),
                    "raw_exposure": raw_exposure,
                    "seed": seed,
                }
            )
            x_train = np.column_stack([np.ones(train_idx.size), train_idx.astype(np.float64) + 1.0])
            x_val = np.column_stack([np.ones(val_idx.size), val_idx.astype(np.float64) + 1.0])
            return x_train, x_val, ["Intercept", "pos(x)"]

        monkeypatch.setattr(reg_path, "build_fold_design_matrices", fake_build_fold_design_matrices)
        return calls

    @pytest.mark.parametrize(
        ("regularization", "l1_ratio", "expected_l1", "expected_type"),
        [
            ("ridge", None, 0.0, "ridge"),
            ("lasso", None, 1.0, "lasso"),
            ("elastic_net", 0.3, 0.3, "elastic_net"),
        ],
    )
    def test_fit_cv_te_regularization_path_handles_fold_failures_and_reg_types(
        self,
        monkeypatch,
        capsys,
        regularization,
        l1_ratio,
        expected_l1,
        expected_type,
    ):
        design_calls = self._mock_te_designs(monkeypatch)
        alpha_calls = []

        def fake_compute_alpha_max(*_args, **_kwargs):
            alpha_calls.append(_kwargs)
            if len(alpha_calls) == 1:
                raise ValidationError("degenerate fold")
            return 10.0

        fold_fit_calls = []

        def fake_fit_fold_path(*args):
            active_alphas = list(args[12])
            fold_fit_calls.append(active_alphas)
            if len(fold_fit_calls) == 1:
                return {"fold_deviances": [1.0, np.inf, 2.0][: len(active_alphas)]}
            return {"fold_deviances": [1.2, 2.0][: len(active_alphas)]}

        monkeypatch.setattr(reg_path, "compute_alpha_max", fake_compute_alpha_max)
        monkeypatch.setattr(rust_backend, "fit_fold_path_py", fake_fit_fold_path)
        glm = SimpleNamespace(
            X=np.ones((4, 2)),
            y=np.array([0.0, 1.0, 2.0, 3.0]),
            family="gaussian",
            link="identity",
            var_power=1.5,
            theta=None,
            offset=np.array([0.0, 0.1, 0.2, 0.3]),
            weights=np.array([1.0, 2.0, 1.0, 2.0]),
            allow_extended_tweedie=True,
            intercept=True,
            _builder=SimpleNamespace(_parsed_formula=SimpleNamespace(label="parsed")),
            data=SimpleNamespace(label="data"),
            _raw_exposure=np.array([1.0, 1.0, 2.0, 2.0]),
        )

        info = fit_cv_te_regularization_path(
            glm,
            cv=2,
            selection="min",
            regularization=regularization,
            n_alphas=2,
            alpha_min_ratio=0.1,
            l1_ratio=l1_ratio,
            max_iter=9,
            tol=1e-6,
            seed=None,
            include_unregularized=True,
            verbose=True,
            standardize=True,
        )

        assert info.fold_safe_target_encoding is True
        assert info.regularization_type == expected_type
        assert info.selected_alpha == pytest.approx(10.0)
        assert info.selected_l1_ratio == pytest.approx(expected_l1)
        assert info.cv_max_iter == 9
        assert info.cv_tol == pytest.approx(1e-6)
        assert info.cv_fold_scores is not None
        assert 1.0 in info.cv_fold_scores
        assert len(info.path) == 2
        assert all(result.n_nonzero == 1 for result in info.path)
        assert design_calls[0]["seed"] == 42
        np.testing.assert_array_equal(design_calls[0]["raw_exposure"], glm._raw_exposure)
        assert "fold alpha_max computation failed" in capsys.readouterr().out

    def test_fit_cv_te_regularization_path_reports_none_when_zero_alpha_wins(
        self,
        monkeypatch,
    ):
        self._mock_te_designs(monkeypatch)
        monkeypatch.setattr(reg_path, "compute_alpha_max", lambda *_args, **_kwargs: 10.0)

        def fake_fit_fold_path(*args):
            active_alphas = list(args[12])
            return {"fold_deviances": [0.5 if alpha == 0.0 else 2.0 for alpha in active_alphas]}

        monkeypatch.setattr(rust_backend, "fit_fold_path_py", fake_fit_fold_path)
        glm = SimpleNamespace(
            X=np.ones((4, 2)),
            y=np.array([0.0, 1.0, 2.0, 3.0]),
            family="gaussian",
            link="identity",
            var_power=1.5,
            theta=2.0,
            offset=None,
            weights=None,
            intercept=True,
            _builder=SimpleNamespace(_parsed_formula=SimpleNamespace()),
            data=SimpleNamespace(),
        )

        info = fit_cv_te_regularization_path(
            glm,
            cv=2,
            regularization="lasso",
            n_alphas=1,
            include_unregularized=True,
            standardize=False,
        )

        assert info.selected_alpha == 0.0
        assert info.regularization_type == "none"
        assert info.path[-1].l1_ratio == 0.0

    def test_fit_cv_te_regularization_path_fails_closed_when_all_fold_fits_fail(
        self,
        monkeypatch,
    ):
        self._mock_te_designs(monkeypatch)
        monkeypatch.setattr(reg_path, "compute_alpha_max", lambda *_args, **_kwargs: 1.0)
        monkeypatch.setattr(
            rust_backend,
            "fit_fold_path_py",
            lambda *_args: (_ for _ in ()).throw(ValueError("fold failed")),
        )
        glm = SimpleNamespace(
            X=np.ones((4, 2)),
            y=np.array([0.0, 1.0, 2.0, 3.0]),
            family="gaussian",
            link="identity",
            var_power=1.5,
            theta=None,
            offset=None,
            weights=None,
            intercept=True,
            _builder=SimpleNamespace(_parsed_formula=SimpleNamespace()),
            data=SimpleNamespace(),
        )

        with pytest.raises(ValidationError, match="no finite validation deviances"):
            fit_cv_te_regularization_path(
                glm,
                cv=2,
                regularization="ridge",
                n_alphas=1,
                include_unregularized=True,
                standardize=False,
            )

        with pytest.raises(ValidationError, match="Unknown regularization"):
            fit_cv_te_regularization_path(glm, regularization="adaptive")
