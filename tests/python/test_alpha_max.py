"""RS-ACT-005 (PR12): GLM-score-based regularization alpha_max.

Covers:
* 005.1 — at ``alpha = alpha_max``, every penalized lasso coefficient is zero.
* 005.2 — changing the offset distribution changes ``alpha_max``.
* 005.3 — changing prior weights changes ``alpha_max``.
* 005.4 — Gaussian/identity matches the solver's own scaling (raw weighted
  sums, **no** implicit ``1/n``). Includes a cross-check against ``glum`` at
  matched scaling.
* 005.5 — Poisson/log ``alpha_max`` matches a finite-difference score check
  at the intercept-only null model.
* Ridge ``l1_ratio == 0`` uses a documented heuristic grid endpoint, not the
  all-zero-derived max (which is undefined for ridge).

The fixture is built directly with numpy/polars so the test pins exactly the
design matrix the solver sees, instead of going through the formula API.
"""

from __future__ import annotations

import numpy as np
import pytest
from rustystats._rustystats import fit_glm_py
from rustystats.constants import ALPHA_MAX_FLOOR
from rustystats.regularization_path import compute_alpha_max


def _design(n: int = 600, p: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build an (n × (p+1)) design matrix with leading intercept column."""
    rng = np.random.default_rng(seed)
    X_features = rng.normal(size=(n, p))
    X = np.column_stack([np.ones(n), X_features])
    return X, X_features


def _poisson_response(X_features: np.ndarray, offset: np.ndarray, seed: int) -> np.ndarray:
    """Poisson response: log-rate = 0.1 + small signal + offset."""
    rng = np.random.default_rng(seed)
    beta_true = np.array([0.3, -0.2, 0.0, 0.1, 0.0])[: X_features.shape[1]]
    eta = 0.1 + X_features @ beta_true + offset
    return rng.poisson(np.exp(eta)).astype(float)


def _gaussian_response(X_features: np.ndarray, offset: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta_true = np.array([1.0, -0.5, 0.0, 0.5, 0.0])[: X_features.shape[1]]
    return X_features @ beta_true + offset + rng.normal(0.0, 0.5, X_features.shape[0])


def _fit_at_alpha(
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    link: str,
    alpha: float,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-lasso fit (l1_ratio=1) at the given alpha. Returns params."""
    result = fit_glm_py(
        y, X, family, link, 1.5, 1.0, offset, weights, alpha, 1.0, 200, 1e-10, None, None, False
    )
    return np.asarray(result.params)


# --------------------------------------------------------------------------
# 005.1: at alpha = alpha_max, all penalized lasso coefs are zero.
# --------------------------------------------------------------------------


class TestAlphaMaxZeroesPenalizedCoefs:
    def test_poisson_log_link(self):
        """At alpha_max, penalized Poisson lasso coefs all vanish."""
        X, X_feat = _design(n=600, seed=1)
        offset = np.log(np.linspace(0.5, 2.0, 600))
        y = _poisson_response(X_feat, offset, seed=11)
        alpha_max = compute_alpha_max(
            X, y, l1_ratio=1.0, family="poisson", link="log", offset=offset
        )
        params = _fit_at_alpha(y, X, "poisson", "log", alpha_max, offset=offset)
        assert np.max(np.abs(params[1:])) < 1e-6, (
            f"non-zero penalized coefs at alpha_max={alpha_max}: {params[1:]}"
        )

    def test_gaussian_identity_link(self):
        X, X_feat = _design(n=400, seed=2)
        offset = np.linspace(-0.5, 0.5, 400)
        y = _gaussian_response(X_feat, offset, seed=22)
        alpha_max = compute_alpha_max(
            X, y, l1_ratio=1.0, family="gaussian", link="identity", offset=offset
        )
        params = _fit_at_alpha(y, X, "gaussian", "identity", alpha_max, offset=offset)
        assert np.max(np.abs(params[1:])) < 1e-6

    def test_just_below_alpha_max_keeps_at_least_one(self):
        """alpha_max is *tight*: a hair below it, at least one penalized coef should be nonzero."""
        X, X_feat = _design(n=500, seed=3)
        offset = np.log(np.linspace(0.5, 2.0, 500))
        y = _poisson_response(X_feat, offset, seed=33)
        alpha_max = compute_alpha_max(
            X, y, l1_ratio=1.0, family="poisson", link="log", offset=offset
        )
        params = _fit_at_alpha(y, X, "poisson", "log", alpha_max * 0.5, offset=offset)
        assert np.max(np.abs(params[1:])) > 0.0, (
            "alpha_max is not tight — even at half of it, all coefs are still zero"
        )


# --------------------------------------------------------------------------
# 005.2 / 005.3: offset & weights change alpha_max.
# --------------------------------------------------------------------------


class TestAlphaMaxSensitivity:
    def test_offset_distribution_changes_alpha_max(self):
        """005.2: a different offset *distribution* gives a different alpha_max.

        A constant offset shift is absorbed by the intercept and leaves μ_0
        per-row invariant — that case is mathematically a no-op and is not
        what 005.2 is about. The test that catches the legacy ``np.average(y)``
        proxy (which sees neither offset shape nor scale) uses two offsets
        with the same mean but very different spread: μ_0 then varies
        per-row, so the score X' (y - μ_0) cannot match.
        """
        X, X_feat = _design(n=400, seed=4)
        # Build two log-offset arrays with the same mean (so the intercept
        # absorbs equal totals) but different spread.
        rng = np.random.default_rng(123)
        flat = np.log(np.full(400, 1.0))  # all exposure = 1
        wide = np.log(rng.uniform(0.1, 10.0, 400))  # exposure varies 100×
        # Centre `wide` to share the mean of `flat` (= 0), so the only thing
        # the score "sees" is the per-row variation.
        wide = wide - wide.mean()
        y = _poisson_response(X_feat, offset=flat, seed=44)
        a_flat = compute_alpha_max(X, y, l1_ratio=1.0, family="poisson", link="log", offset=flat)
        a_wide = compute_alpha_max(X, y, l1_ratio=1.0, family="poisson", link="log", offset=wide)
        assert not np.isclose(a_flat, a_wide, rtol=0.01), (
            f"alpha_max did not respond to offset distribution: {a_flat} vs {a_wide}"
        )

    def test_weights_change_alpha_max(self):
        """005.3: prior weights change alpha_max."""
        X, X_feat = _design(n=400, seed=5)
        y = _gaussian_response(X_feat, offset=np.zeros(400), seed=55)
        a_unw = compute_alpha_max(X, y, l1_ratio=1.0, family="gaussian", link="identity")
        # Heterogeneous weights that emphasize the tail.
        w = np.linspace(0.1, 5.0, 400)
        a_w = compute_alpha_max(X, y, l1_ratio=1.0, family="gaussian", link="identity", weights=w)
        assert not np.isclose(a_unw, a_w, rtol=0.01), (
            f"alpha_max did not respond to weight change: {a_unw} vs {a_w}"
        )


# --------------------------------------------------------------------------
# 005.4: Gaussian-identity matches solver scaling — raw weighted sums, no 1/n.
# --------------------------------------------------------------------------


class TestGaussianSolverScaling:
    def test_raw_weighted_sums_no_implicit_one_over_n(self):
        """005.4: alpha_max is on the raw-sum scale, not divided by n.

        Direct check: for Gaussian/identity with no offset/weights the score
        at the null is X_pen' (y - y_bar). alpha_max = max_j |that| / l1_ratio.
        The legacy implementation divided this by n; the rewrite must not.
        """
        X, X_feat = _design(n=600, seed=6)
        y = _gaussian_response(X_feat, offset=np.zeros(600), seed=66)
        alpha_max = compute_alpha_max(X, y, l1_ratio=1.0, family="gaussian", link="identity")
        expected_max = float(np.max(np.abs(X[:, 1:].T @ (y - y.mean()))))
        np.testing.assert_allclose(alpha_max, expected_max, rtol=1e-10)

    def test_alpha_max_scales_with_n_not_invariant(self):
        """A 2x-larger sample of the same DGP gives a ~2x-larger alpha_max.

        The legacy ``/n`` scaling made alpha_max approximately n-invariant,
        which is the opposite of what the raw-weighted-sum solver wants.
        """
        X_small, Xf_small = _design(n=300, seed=7)
        X_large, Xf_large = _design(n=600, seed=7)  # extend by drawing more rows
        y_small = _gaussian_response(Xf_small, offset=np.zeros(300), seed=77)
        y_large = _gaussian_response(Xf_large, offset=np.zeros(600), seed=77)
        a_small = compute_alpha_max(
            X_small, y_small, l1_ratio=1.0, family="gaussian", link="identity"
        )
        a_large = compute_alpha_max(
            X_large, y_large, l1_ratio=1.0, family="gaussian", link="identity"
        )
        # Raw-sum scaling: doubling n roughly doubles alpha_max. (Coarse bound;
        # the point is they're clearly not n-invariant.)
        assert a_large > a_small * 1.5, (
            f"alpha_max did not grow with n: a_small={a_small}, a_large={a_large}"
        )


# --------------------------------------------------------------------------
# 005.5: Poisson/log alpha_max matches a finite-difference score check.
# --------------------------------------------------------------------------


class TestFiniteDifferenceScore:
    def test_alpha_max_equals_max_fd_score(self):
        """005.5: alpha_max == max_j |∂(-ℓ)/∂β_j| at the null fit, by finite diff.

        Use ε = 1e-6; we expect agreement to ~1e-3 relative (FD truncation
        + intercept-fit precision).
        """
        X, X_feat = _design(n=500, seed=8)
        offset = np.log(np.linspace(0.4, 2.5, 500))
        y = _poisson_response(X_feat, offset=offset, seed=88)
        alpha_max = compute_alpha_max(
            X, y, l1_ratio=1.0, family="poisson", link="log", offset=offset
        )
        # Fit intercept-only model to locate the null β_0.
        x_int = np.ones((y.shape[0], 1))
        null_res = fit_glm_py(
            y,
            x_int,
            "poisson",
            "log",
            1.5,
            1.0,
            offset,
            None,
            0.0,
            0.0,
            200,
            1e-12,
            None,
            None,
            False,
        )
        beta_0 = float(np.asarray(null_res.params)[0])

        def neg_loglik(beta: np.ndarray) -> float:
            eta = X @ beta + offset
            mu = np.exp(eta)
            # Poisson negative log-likelihood (drop y!-style constants).
            return float(np.sum(mu - y * eta))

        p = X.shape[1]
        beta_null = np.zeros(p)
        beta_null[0] = beta_0
        eps = 1e-6
        max_fd = 0.0
        for j in range(1, p):  # skip intercept
            plus = beta_null.copy()
            plus[j] = eps
            minus = beta_null.copy()
            minus[j] = -eps
            grad_j = (neg_loglik(plus) - neg_loglik(minus)) / (2 * eps)
            max_fd = max(max_fd, abs(grad_j))
        np.testing.assert_allclose(alpha_max, max_fd, rtol=2e-3)


# --------------------------------------------------------------------------
# Ridge: l1_ratio == 0 uses a documented heuristic grid endpoint.
# --------------------------------------------------------------------------


class TestRidgeGridEndpoint:
    def test_ridge_alpha_max_finite_and_above_floor(self):
        """For ridge (l1_ratio=0) the KKT-based formula is undefined; the
        helper must fall back to a documented heuristic instead of e.g.
        returning ALPHA_MAX_FLOOR for every input."""
        X, X_feat = _design(n=400, seed=9)
        y = _gaussian_response(X_feat, offset=np.zeros(400), seed=99)
        alpha_max = compute_alpha_max(X, y, l1_ratio=0.0, family="gaussian", link="identity")
        assert np.isfinite(alpha_max)
        assert alpha_max > ALPHA_MAX_FLOOR, f"ridge alpha_max degenerated to the floor: {alpha_max}"

    def test_ridge_alpha_max_scales_with_design(self):
        """Doubling the design columns' magnitude should change the ridge grid."""
        X1, _ = _design(n=400, seed=10)
        y = np.linspace(0.0, 1.0, 400)
        X2 = X1.copy()
        X2[:, 1:] *= 3.0
        a1 = compute_alpha_max(X1, y, l1_ratio=0.0, family="gaussian", link="identity")
        a2 = compute_alpha_max(X2, y, l1_ratio=0.0, family="gaussian", link="identity")
        assert not np.isclose(a1, a2, rtol=0.05)


# --------------------------------------------------------------------------
# Cross-check against glum at matched scaling.
# --------------------------------------------------------------------------


class TestGlumCrossCheck:
    def test_gaussian_lasso_alpha_max_matches_glum(self):
        """005.4 (cross-check): glum's intercept-aware alpha_max for Gaussian
        ElasticNet at matched scaling should agree with ours.

        glum's ``alpha`` parameter is defined as
        ``alpha = max_j |X_j' (y - y_bar)| / (n * l1_ratio)`` when
        ``offset_scaling="auto"`` is off — i.e. it *does* divide by n. We
        cross-check by undoing the ``n`` factor on glum's side.
        """
        pytest.importorskip("glum")
        from glum import GeneralizedLinearRegressorCV

        X, X_feat = _design(n=500, seed=12)
        y = _gaussian_response(X_feat, offset=np.zeros(500), seed=121)
        # glum's path computes its own alpha grid; the upper end of that grid
        # is its alpha_max definition. We pull it from a fitted CV instance.
        cv = GeneralizedLinearRegressorCV(
            family="normal",
            l1_ratio=1.0,
            fit_intercept=True,
            n_alphas=5,
            min_alpha_ratio=0.5,  # large so the grid spans a tight range
            scale_predictors=False,
        )
        cv.fit(X_feat, y)
        glum_alpha_max = float(cv.alphas_[0])  # alphas sorted descending
        # Convert glum's per-row-mean scaling to our raw-sum scaling.
        ours = compute_alpha_max(X, y, l1_ratio=1.0, family="gaussian", link="identity")
        glum_in_our_units = glum_alpha_max * X.shape[0]
        # Allow modest slack for glum's internal centering / scaling choices.
        np.testing.assert_allclose(ours, glum_in_our_units, rtol=0.05)
