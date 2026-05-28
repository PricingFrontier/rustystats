"""RS-ACT-001: fold-safe target-encoding CV, explicit convergence, weighted scoring.

PR5 scope (RS-ACT-001b): cross-validated regularization with a target-encoded
term must be *fold-safe* -- the encoding is fit on each fold's training rows
only and applied to the held-out rows, so no validation targets leak into alpha
selection. This replaces the PR3 fail-closed guard, which raised instead.

Non-target-encoded models keep using the fast Rust array CV path unchanged.
CV fold fits must use the requested convergence settings, and validation scoring
must be weighted.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _fixtures import RARE_BRAND, make_freq_frame

CV_KW = {"cv": 3, "regularization": "ridge", "n_alphas": 5, "verbose": False}


def _te_col(names: list[str], needle: str = "Brand") -> int:
    """Index of the single target-encoding design column for ``needle``."""
    cols = [i for i, nm in enumerate(names) if needle in nm]
    assert len(cols) == 1, f"expected one {needle} column, got {cols} in {names}"
    return cols[0]


class TestFoldSafeDesignMatrices:
    """Unit-level guarantees for the per-fold fit/transform split (RS-ACT-001b)."""

    def _frame_with_isolated_rare_level(self):
        """A frame whose RARE rows carry an extreme rate and sit only in the val fold."""
        from rustystats.formula import dict_to_parsed_formula

        rng = np.random.default_rng(0)
        n = 600
        n_rare = 10
        brand = np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)]
        y = rng.poisson(1.0, n).astype(float)
        # Inject a rare level with a wildly different (extreme) rate.
        brand[:n_rare] = RARE_BRAND
        y[:n_rare] = 50.0
        data = pl.DataFrame({"y": y, "Brand": brand})
        parsed = dict_to_parsed_formula(
            response="y",
            terms={"Brand": {"type": "target_encoding"}},
            interactions=None,
            intercept=True,
        )
        # All RARE rows land in validation; training never sees the level.
        val_idx = np.arange(20)
        train_idx = np.arange(20, n)
        return data, parsed, train_idx, val_idx, n_rare

    def test_rare_level_isolated_to_validation_fold_is_not_leaked(self):
        """001.1: a level held out of training gets the fold-training prior, not its own rate."""
        from rustystats.interactions import InteractionBuilder
        from rustystats.regularization_path import build_fold_design_matrices

        data, parsed, train_idx, val_idx, n_rare = self._frame_with_isolated_rare_level()

        _x_train, x_val, names = build_fold_design_matrices(data, parsed, train_idx, val_idx)
        te_col = _te_col(names)

        # The RARE rows are the first n_rare entries of val_idx.
        safe = x_val[:n_rare, te_col]
        # Unseen-in-training -> every RARE row collapses to the same training prior.
        np.testing.assert_allclose(safe, safe[0])

        # The leaky one-shot encoding (fit on full data) sees y=50 for RARE.
        full = InteractionBuilder(data)
        _, x_full, full_names = full.build_design_matrix_from_parsed(parsed)
        leaky = x_full[:n_rare, _te_col(full_names)]
        # Fold-safe encoding must differ from -- and be far below -- the leaked rate.
        assert not np.isclose(safe[0], leaky[0])
        assert safe[0] < leaky[0]

    def test_exposure_weighted_te_in_fold_uses_training_exposure_only(self):
        """001.5: exposure-weighted TE in a fold uses fold-training claims & exposure only."""
        from rustystats.regularization_path import build_fold_design_matrices

        data, parsed, train_idx, val_idx, n_rare = self._frame_with_isolated_rare_level()
        rng = np.random.default_rng(1)
        exposure = rng.uniform(0.5, 1.5, data.height)

        _x_train, x_val, names = build_fold_design_matrices(
            data, parsed, train_idx, val_idx, raw_exposure=exposure
        )
        # Unseen level still collapses to the (exposure-weighted) training prior.
        safe = x_val[:n_rare, _te_col(names)]
        np.testing.assert_allclose(safe, safe[0])

    def test_full_fold_reproduces_one_shot_design(self):
        """(R) the helper with train=all rows reproduces the one-shot build (behaviour-preserving).

        Fit-time target encoding is cross-fitted per row (out-of-fold), so it does
        *not* equal the transform-time encoding -- that asymmetry is by design. The
        guarantee we pin is that building a fold whose training set is the whole
        frame yields the same design matrix as the existing one-shot path.
        """
        from rustystats.interactions import InteractionBuilder
        from rustystats.regularization_path import build_fold_design_matrices

        data, parsed, _train_idx, _val_idx, _n_rare = self._frame_with_isolated_rare_level()
        all_idx = np.arange(data.height)
        x_train, _x_val, _names = build_fold_design_matrices(
            data, parsed, all_idx, all_idx[:5], seed=0
        )

        builder = InteractionBuilder(data)
        _, x_one_shot, _ = builder.build_design_matrix_from_parsed(parsed, seed=0)
        np.testing.assert_allclose(x_train, x_one_shot, rtol=1e-12, atol=1e-12)


class TestFoldSafeTargetEncodingCV:
    """The PR3 guard is gone: CV + target encoding now runs fold-safe."""

    def test_cv_with_target_encoding_term_runs(self):
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        result = model.fit(**CV_KW)
        assert result.cv_deviance is not None
        assert np.isfinite(result.cv_deviance)
        assert result.converged

    def test_cv_with_target_encoding_interaction_runs(self):
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"Region": {"type": "categorical"}, "Brand": {"type": "categorical"}},
            interactions=[
                {
                    "Region": {"type": "categorical"},
                    "Brand": {"type": "categorical"},
                    "target_encoding": True,
                }
            ],
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        result = model.fit(**CV_KW)
        assert result.cv_deviance is not None
        assert np.isfinite(result.cv_deviance)

    def test_cv_with_target_encoded_factor_inside_standard_interaction_runs(self):
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}},
            interactions=[
                {
                    "DrivAge": {"type": "linear"},
                    "Brand": {"type": "target_encoding"},
                }
            ],
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        result = model.fit(**CV_KW)
        assert result.cv_deviance is not None
        assert np.isfinite(result.cv_deviance)

    def test_cv_te_is_reproducible(self):
        """Same seed -> identical selected alpha and CV deviance."""
        df = make_freq_frame()

        def _fit():
            return rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
                data=df,
                family="poisson",
                exposure="Exposure",
                seed=7,
            ).fit(cv=3, regularization="ridge", n_alphas=5, cv_seed=7, verbose=False)

        r1, r2 = _fit(), _fit()
        np.testing.assert_array_equal(r1.params, r2.params)
        assert r1.cv_deviance == pytest.approx(r2.cv_deviance)

    def test_cv_te_alpha_grid_uses_fold_training_designs(self, monkeypatch):
        """001.2: alpha candidates are derived after the fold split, not from full-data TE."""
        from rustystats import regularization_path as rp

        df = make_freq_frame()
        seen_rows = []
        original = rp.compute_alpha_max

        def spy_compute_alpha_max(X, y, l1_ratio, **kwargs):
            seen_rows.append(X.shape[0])
            return original(X, y, l1_ratio, **kwargs)

        monkeypatch.setattr(rp, "compute_alpha_max", spy_compute_alpha_max)
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit(cv=3, regularization="ridge", n_alphas=3, cv_seed=7, verbose=False)

        assert result.cv_deviance is not None
        assert len(seen_rows) == 3
        assert all(n_rows < df.height for n_rows in seen_rows)

    def test_cv_te_fold_fits_receive_sign_constraints(self, monkeypatch):
        import rustystats._rustystats as rust

        rng = np.random.default_rng(0)
        n = 180
        x = rng.uniform(0, 5, n)
        brand = rng.choice(["A", "B", "C", "D"], n)
        y = rng.poisson(np.exp(0.2 + 0.1 * x)).astype(float)
        data = pl.DataFrame({"y": y, "x": x, "Brand": brand})

        original = rust.fit_glm_py
        seen_nonneg = []

        def spy_fit_glm_py(*args, **kwargs):
            if len(args) >= 13:
                seen_nonneg.append(args[12])
            else:
                seen_nonneg.append(kwargs.get("nonneg_indices"))
            return original(*args, **kwargs)

        monkeypatch.setattr(rust, "fit_glm_py", spy_fit_glm_py)
        result = rs.glm_dict(
            response="y",
            terms={
                "x": {"type": "linear", "monotonicity": "increasing"},
                "Brand": {"type": "target_encoding"},
            },
            data=data,
            family="poisson",
        ).fit(cv=3, regularization="ridge", n_alphas=3, cv_seed=7, verbose=False)

        assert result.cv_deviance is not None
        assert sum(bool(indices) for indices in seen_nonneg) > 1

    def test_cv_te_drops_alpha_when_any_fold_fit_fails(self, monkeypatch):
        import rustystats._rustystats as rust

        df = make_freq_frame()
        original = rust.fit_glm_py
        failed = {"alpha": None}

        def spy_fit_glm_py(*args, **kwargs):
            alpha = float(args[8])
            if failed["alpha"] is None and alpha > 0:
                failed["alpha"] = alpha
                raise ValueError("forced fold failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(rust, "fit_glm_py", spy_fit_glm_py)
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit(cv=3, regularization="ridge", n_alphas=3, cv_seed=7, verbose=False)

        assert failed["alpha"] is not None
        assert all(row["alpha"] != failed["alpha"] for row in result.regularization_path)

    def test_cv_without_target_encoding_still_works(self):
        """001.5: non-TE models keep working (fast Rust array path)."""
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        result = model.fit(**CV_KW)
        assert result.cv_deviance is not None
        assert result.converged


class TestCVExplicitConvergence:
    def test_cv_uses_requested_convergence_not_relaxed(self):
        """001.6: CV fold fits use the requested max_iter/tol, not hidden 10/1e-4."""
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        result = model.fit(max_iter=50, tol=1e-7, **CV_KW)
        conv = result.cv_convergence
        assert conv is not None
        assert conv["max_iter"] == 50  # not capped at 10
        assert conv["tol"] == pytest.approx(1e-7)  # not relaxed to 1e-4


class TestCVWeightedScoring:
    def test_compute_deviance_uses_validation_weights(self):
        from rustystats._rustystats import PoissonFamily
        from rustystats.regularization_path import compute_deviance

        y = np.array([0.0, 3.0, 10.0])
        mu = np.array([1.0, 1.0, 1.0])
        weights = np.array([10.0, 1.0, 1.0])

        unit_dev = PoissonFamily().unit_deviance(y, mu)
        expected = np.sum(weights * unit_dev) / np.sum(weights)

        assert compute_deviance(y, mu, "poisson", weights=weights) == pytest.approx(expected)
        assert compute_deviance(y, mu, "poisson", weights=weights) != pytest.approx(
            np.mean(unit_dev)
        )
