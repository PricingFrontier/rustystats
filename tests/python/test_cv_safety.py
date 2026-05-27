"""RS-ACT-001a: CV fail-closed guard and explicit convergence.

PR3 scope: until fold-safe target-encoding CV lands (RS-ACT-001b / PR5),
cross-validated regularization with a target-encoded term must fail closed
rather than silently leak validation targets into model selection. CV fold fits
must also use the requested convergence settings, not hidden relaxed ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import rustystats as rs
from _fixtures import make_freq_frame

CV_KW = {"cv": 3, "regularization": "ridge", "n_alphas": 5, "verbose": False}


class TestCVTargetEncodingGuard:
    def test_cv_with_target_encoding_term_raises(self):
        df = make_freq_frame()
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        )
        with pytest.raises(rs.ValidationError, match=r"(?i)fold-safe|target"):
            model.fit(**CV_KW)

    def test_cv_with_target_encoding_interaction_raises(self):
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
        with pytest.raises(rs.ValidationError, match=r"(?i)fold-safe|target"):
            model.fit(**CV_KW)

    def test_cv_with_target_encoded_factor_inside_standard_interaction_raises(self):
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
        with pytest.raises(rs.ValidationError, match=r"(?i)fold-safe|target"):
            model.fit(**CV_KW)

    def test_cv_without_target_encoding_still_works(self):
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
