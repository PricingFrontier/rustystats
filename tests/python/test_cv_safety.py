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

    def test_cv_score_changes_when_validation_weights_change(self):
        """001.4 (end-to-end): heterogeneous validation weights move the CV score.

        Fits the same data with uniform vs heterogeneous prior weights and
        asserts the resulting CV deviances differ. ``compute_deviance`` is
        unit-tested above; this test asserts the *path* threads weights through
        the validation scorer end-to-end, not just the helper.
        """
        rng = np.random.default_rng(7)
        n = 600
        x = rng.normal(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.2 + 0.3 * x)).astype(float)
        df = pl.DataFrame({"y": y, "x": x})

        # Heterogeneous weights that emphasise the tail.
        het_w = np.linspace(0.1, 5.0, n)

        def _fit(weights):
            return rs.glm_dict(
                response="y",
                terms={"x": {"type": "linear"}},
                data=df.with_columns(pl.Series("w", weights)),
                family="poisson",
                weights="w",
            ).fit(cv=3, regularization="ridge", n_alphas=5, cv_seed=11, verbose=False)

        r_uniform = _fit(np.ones(n))
        r_hetero = _fit(het_w)

        # Either the selected alpha or the CV deviance must move; if both are
        # unchanged the validation weights are not being used in scoring.
        moved_alpha = not np.isclose(r_uniform.alpha, r_hetero.alpha)
        moved_deviance = not np.isclose(r_uniform.cv_deviance, r_hetero.cv_deviance, rtol=1e-6)
        assert moved_alpha or moved_deviance, (
            "CV score unchanged by validation weights: "
            f"uniform α={r_uniform.alpha} dev={r_uniform.cv_deviance}, "
            f"hetero α={r_hetero.alpha} dev={r_hetero.cv_deviance}"
        )

    def test_weighted_cv_score_is_normalized_by_sum_of_weights(self):
        """001.4 (Rust array-path oracle): the scorer computes Σ(w·dev)/Σw, not Σ(w·dev).

        Calls ``fit_cv_path_py`` directly with a fixed *unpenalized* alpha so each
        fold's fit is invariant to a uniform weight rescale. A correct ``/Σw``
        normaliser then makes a uniform weight cancel (scores unchanged), whereas
        a dropped denominator would scale every fold score by the weight.
        Heterogeneous weights must still move the score (weights reach the
        scorer). The end-to-end test above only checks "something moved"; this
        pins the normalisation that the wrong-denominator class of bug would break.
        """
        from rustystats._rustystats import fit_cv_path_py

        rng = np.random.default_rng(0)
        n, p = 240, 3
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p))])
        offset = np.log(np.linspace(0.5, 2.0, n))
        eta = 0.2 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + offset
        y = rng.poisson(np.exp(eta)).astype(float)

        common = {
            "family": "poisson",
            "link": "log",
            "offset": offset,
            "alphas": [0.0],  # unpenalized -> fit invariant to uniform weight rescale
            "l1_ratio": 0.0,
            "n_folds": 4,
            "seed": 7,
        }
        base = fit_cv_path_py(y, X, weights=None, **common)
        uniform = fit_cv_path_py(y, X, weights=np.full(n, 5.0), **common)
        hetero = fit_cv_path_py(y, X, weights=rng.uniform(0.2, 3.0, n), **common)

        # Uniform weight cancels in Σ(w·dev)/Σw -> per-fold scores unchanged.
        np.testing.assert_allclose(
            base["cv_fold_scores"][0], uniform["cv_fold_scores"][0], rtol=1e-9
        )
        # ...but heterogeneous weights genuinely change the score.
        assert not np.allclose(base["cv_fold_scores"][0], hetero["cv_fold_scores"][0])

    def test_validation_deviance_excludes_regularization_penalty(self):
        """001.7: validation deviance is unit deviance only — the α·||β||² penalty
        must NOT contribute to the CV score.

        Constructs a small ridge fit (routed through the fold-safe Python path
        via a TE term so we can reproduce the fold split with
        :func:`create_cv_folds`), recovers per-fold β from a manual CV
        reproduction, hand-computes the weighted-mean validation unit
        deviance, and asserts it matches the reported ``cv_deviance``. A leak
        would show up as ``cv_deviance ≈ unit_dev + α||β||²`` (strictly larger).
        """
        from rustystats._rustystats import fit_glm_py
        from rustystats.formula import dict_to_parsed_formula
        from rustystats.regularization_path import (
            build_fold_design_matrices,
            compute_deviance,
            create_cv_folds,
        )

        rng = np.random.default_rng(101)
        n = 300
        x1 = rng.normal(0.0, 1.0, n)
        brand = rng.choice(["A", "B", "C", "D"], n)
        # Larger-than-typical signal so β is well away from zero and the
        # penalty contribution would be detectable if it leaked.
        y = rng.poisson(np.exp(0.3 + 0.4 * x1)).astype(float)
        df = pl.DataFrame({"y": y, "x1": x1, "Brand": brand})

        cv_seed = 13
        n_folds = 3
        terms = {"x1": {"type": "linear"}, "Brand": {"type": "target_encoding"}}
        result = rs.glm_dict(
            response="y",
            terms=terms,
            data=df,
            family="poisson",
        ).fit(cv=n_folds, regularization="ridge", n_alphas=4, cv_seed=cv_seed, verbose=False)

        # Pick the smallest non-zero alpha actually scored on the path — α=0
        # has no penalty so it cannot detect a leak.
        non_zero = [p for p in result.regularization_path if p["alpha"] > 0]
        assert non_zero, "no penalised alpha was scored on the path"
        row = min(non_zero, key=lambda p: p["cv_deviance_mean"])
        selected_alpha = row["alpha"]
        expected_dev = row["cv_deviance_mean"]

        parsed = dict_to_parsed_formula(
            response="y", terms=terms, interactions=None, intercept=True
        )
        folds = create_cv_folds(n, n_folds, cv_seed)
        fold_devs: list[float] = []
        fold_penalty_contributions: list[float] = []
        for train_idx, val_idx in folds:
            x_train, x_val, _names = build_fold_design_matrices(
                df, parsed, train_idx, val_idx, raw_exposure=None, seed=cv_seed
            )
            fit_res = fit_glm_py(
                y[train_idx],
                x_train,
                "poisson",
                "log",
                1.5,  # var_power
                1.0,  # theta
                None,  # offset
                None,  # weights
                selected_alpha,
                0.0,  # ridge: l1_ratio = 0
                25,  # max_iter (matches DEFAULT_MAX_ITER)
                1e-8,  # tol (matches DEFAULT_TOLERANCE)
                None,
                None,
                False,
            )
            beta = np.asarray(fit_res.params)
            mu_val = np.exp(x_val @ beta)
            fold_devs.append(compute_deviance(y[val_idx], mu_val, "poisson"))
            # The penalty contribution (alpha * ||beta_penalised||^2) that
            # would leak into the score if the validation deviance accidentally
            # included it. Intercept (index 0) is unpenalised.
            fold_penalty_contributions.append(selected_alpha * float(np.sum(beta[1:] ** 2)))

        manual_mean_dev = float(np.mean(fold_devs))
        mean_penalty = float(np.mean(fold_penalty_contributions))

        # 1. Hand-computed validation unit deviance matches the reported score.
        assert manual_mean_dev == pytest.approx(expected_dev, rel=1e-3), (
            f"manual unit-dev CV score {manual_mean_dev} != reported {expected_dev}"
        )
        # 2. If the penalty had leaked, manual+penalty would be the reported
        #    score; assert that's clearly NOT the case.
        leaked_score = manual_mean_dev + mean_penalty
        assert mean_penalty > 1e-6, (
            "test cannot prove penalty exclusion if penalty contribution is ~0"
        )
        assert not np.isclose(leaked_score, expected_dev, rtol=1e-3), (
            f"reported CV deviance {expected_dev} matches penalty-leaked value "
            f"{leaked_score}; the α·||β||² term may be in the score"
        )


class TestManualFoldCV:
    """RS-ACT-001b (PR5): hand-computed K-fold CV matches the fold-safe path
    on a tiny TE fixture. Pins that the abstraction reproduces the textbook
    operation it claims to."""

    def test_manual_kfold_matches_fold_safe_te_cv(self):
        """001.2 (manual): hand-rolled CV reproduces ``cv_deviance`` on a TE fit.

        Replays the fold-safe path step by step (KFold split -> per-fold
        ``build_fold_design_matrices`` -> per-fold IRLS -> manual mean unit
        deviance) and checks it matches the result the high-level API returns
        for the same seed/grid.
        """
        from rustystats._rustystats import fit_glm_py
        from rustystats.regularization_path import (
            build_fold_design_matrices,
            compute_deviance,
            create_cv_folds,
        )

        rng = np.random.default_rng(31)
        n = 200
        x = rng.uniform(0.0, 5.0, n)
        brand = rng.choice(["A", "B", "C", "D"], n)
        y = rng.poisson(np.exp(0.2 + 0.1 * x)).astype(float)
        df = pl.DataFrame({"y": y, "x": x, "Brand": brand})

        cv_seed = 17
        n_folds = 3
        n_alphas = 4

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
        ).fit(
            cv=n_folds,
            regularization="ridge",
            n_alphas=n_alphas,
            cv_seed=cv_seed,
            verbose=False,
        )

        # Replay the manual CV at the selected alpha (or, if that's 0, the
        # best penalised alpha — α=0 takes a different code path).
        candidates = [p for p in result.regularization_path if p["alpha"] > 0]
        assert candidates, "no penalised alpha was scored"
        target = min(candidates, key=lambda p: p["cv_deviance_mean"])
        target_alpha = target["alpha"]
        expected_dev = target["cv_deviance_mean"]

        from rustystats.formula import dict_to_parsed_formula

        parsed = dict_to_parsed_formula(
            response="y",
            terms={"x": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            interactions=None,
            intercept=True,
        )

        folds = create_cv_folds(n, n_folds, cv_seed)
        fold_devs = []
        for train_idx, val_idx in folds:
            x_train, x_val, _names = build_fold_design_matrices(
                df, parsed, train_idx, val_idx, raw_exposure=None, seed=cv_seed
            )
            fit_res = fit_glm_py(
                y[train_idx],
                x_train,
                "poisson",
                "log",
                1.5,
                1.0,
                None,
                None,
                target_alpha,
                0.0,
                25,
                1e-8,
                None,
                None,
                False,
            )
            beta = np.asarray(fit_res.params)
            mu_val = np.exp(x_val @ beta)
            fold_devs.append(compute_deviance(y[val_idx], mu_val, "poisson"))

        manual_mean = float(np.mean(fold_devs))
        assert manual_mean == pytest.approx(expected_dev, rel=1e-3), (
            f"manual fold CV {manual_mean} != fold-safe path {expected_dev}"
        )


class TestInteractionTELeakage:
    """RS-ACT-001b: 2-way TE interaction must not leak validation targets
    through the *interaction* level encoding either."""

    def test_held_out_interaction_level_uses_training_prior(self):
        """001.3: a (Region, Brand) interaction level held out of training rows
        gets the training prior, not a leaked rate.

        Constructs a frame where a particular interaction cell only appears in
        the validation fold, and asserts that the fold-safe design encodes
        that cell with the same training prior as any other unseen cell
        (so all held-out rows in that cell collapse to one value) — the
        leaky one-shot build would encode it with its own y, instead.
        """
        from rustystats.formula import dict_to_parsed_formula
        from rustystats.interactions import InteractionBuilder
        from rustystats.regularization_path import build_fold_design_matrices

        rng = np.random.default_rng(0)
        n = 600
        # Two factors; the (R_X, B_X) cell only exists in the first 10 rows
        # (the val fold), with an extreme y. All other cells are common.
        region = rng.choice(["A", "B"], n).astype(object)
        brand = rng.choice(["P", "Q"], n).astype(object)
        y = rng.poisson(1.0, n).astype(float)
        # Held-out interaction cell:
        n_rare = 10
        region[:n_rare] = "R_X"
        brand[:n_rare] = "B_X"
        y[:n_rare] = 50.0

        data = pl.DataFrame({"y": y, "Region": region, "Brand": brand})
        parsed = dict_to_parsed_formula(
            response="y",
            terms={
                "Region": {"type": "categorical"},
                "Brand": {"type": "categorical"},
            },
            interactions=[
                {
                    "Region": {"type": "categorical"},
                    "Brand": {"type": "categorical"},
                    "target_encoding": True,
                }
            ],
            intercept=True,
        )
        val_idx = np.arange(n_rare)
        train_idx = np.arange(n_rare, n)

        _x_train, x_val, names = build_fold_design_matrices(data, parsed, train_idx, val_idx)
        te_cols = [i for i, nm in enumerate(names) if "Region" in nm and "Brand" in nm]
        assert te_cols, f"no TE interaction column in design: {names}"
        te_col = te_cols[0]

        held_out_rows = x_val[:n_rare, te_col]
        # Every held-out (R_X, B_X) row gets the same training prior because
        # the level is unseen — fold-safe encoding cannot tell them apart.
        np.testing.assert_allclose(held_out_rows, held_out_rows[0])

        # The leaky full-data encoding would assign a much larger value to
        # these rows because their y is 50, not ~1.
        full_builder = InteractionBuilder(data)
        _, x_full, full_names = full_builder.build_design_matrix_from_parsed(parsed)
        full_te_cols = [i for i, nm in enumerate(full_names) if "Region" in nm and "Brand" in nm]
        leaked = x_full[:n_rare, full_te_cols[0]]
        assert not np.isclose(held_out_rows[0], leaked[0]), (
            "fold-safe encoding matched the leaky full-data encoding: "
            f"safe={held_out_rows[0]} leaky={leaked[0]}"
        )
        assert held_out_rows[0] < leaked[0]


class TestFoldDesignCaching:
    """RS-ACT-001b (PR5 review): the fold-safe path builds each fold's design
    once for the alpha grid and reuses it for the fit pass — not twice."""

    def test_fold_design_built_once_per_fold(self, monkeypatch):
        """001 (perf): builder calls equal exactly cv (n_folds), not 2 * cv."""
        from rustystats import regularization_path as rp

        df = make_freq_frame()
        call_count = {"n": 0}
        original = rp.build_fold_design_matrices

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(rp, "build_fold_design_matrices", spy)
        n_folds = 3
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit(cv=n_folds, regularization="ridge", n_alphas=3, cv_seed=7, verbose=False)

        assert result.cv_deviance is not None
        # Before the cache: 2 * n_folds (alpha-grid pass + fit pass). After:
        # exactly n_folds.
        assert call_count["n"] == n_folds, (
            f"expected {n_folds} builder calls (one per fold), got {call_count['n']}"
        )


class TestFoldSafeDefaultSeed:
    """RS-ACT-001b (PR5 review): the fold-safe path must default to
    ``DEFAULT_CV_SEED`` when ``cv_seed`` is None — same convention as the fast
    Rust array path. A ``None`` propagated to ``create_cv_folds`` would seed
    from OS entropy and break run-to-run reproducibility."""

    def test_default_cv_seed_is_reproducible_without_explicit_seed(self):
        df = make_freq_frame()

        def _fit():
            # No cv_seed= passed; relies on the default.
            return rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
                data=df,
                family="poisson",
                exposure="Exposure",
            ).fit(cv=3, regularization="ridge", n_alphas=4, verbose=False)

        r1, r2 = _fit(), _fit()
        assert r1.alpha == pytest.approx(r2.alpha)
        assert r1.cv_deviance == pytest.approx(r2.cv_deviance)
