"""Direct contracts for GLM summary formatting helpers."""

from __future__ import annotations

import numpy as np
import pytest
from rustystats.exceptions import FittingError, ValidationError
from rustystats.glm import _normal_two_sided_p, _significance_code, summary, summary_relativities


class _FakeGLMResult:
    def __init__(self, *, fail_diagnostics: bool = False):
        self.params = np.array([0.0, 0.25])
        self.family = "poisson"
        self.nobs = 100
        self.df_resid = 98
        self.df_model = 1
        self.deviance = 12.5
        self.converged = True
        self.iterations = 6
        self.is_regularized = False
        self.fail_diagnostics = fail_diagnostics

    def bse(self):
        return np.array([0.1, 0.2])

    def tvalues(self):
        return np.array([0.0, 1.25])

    def pvalues(self):
        return np.array([0.6, 0.02])

    def conf_int(self, alpha):
        assert alpha == 0.05
        return np.array([[-0.2, 0.2], [-0.1, 0.6]])

    def significance_codes(self):
        return ["", "*"]

    def llf(self):
        if self.fail_diagnostics:
            raise RuntimeError("boom")
        return -20.0

    def pearson_chi2(self):
        return 10.0

    def null_deviance(self):
        return 25.0

    def scale(self):
        return 1.0

    def aic(self):
        return 44.0

    def bic(self):
        return 50.0


def test_significance_helpers_cover_threshold_contracts():
    assert _normal_two_sided_p(0.0) == pytest.approx(1.0)
    assert _significance_code(0.0005) == "***"
    assert _significance_code(0.005) == "**"
    assert _significance_code(0.04) == "*"
    assert _significance_code(0.08) == "."
    assert _significance_code(0.2) == ""


def test_summary_defaults_feature_names_and_validates_length():
    result = _FakeGLMResult()

    text = summary(result)

    assert "x0" in text
    assert "x1" in text
    assert "Signif. codes" in text
    with pytest.raises(ValidationError, match="feature_names has 1 elements"):
        summary(result, feature_names=["only_intercept"])


def test_summary_wraps_diagnostic_failures():
    with pytest.raises(FittingError, match="Failed to compute model summary diagnostics"):
        summary(_FakeGLMResult(fail_diagnostics=True))


def test_summary_relativities_defaults_feature_names_and_validates_length():
    result = _FakeGLMResult()

    text = summary_relativities(result)

    assert "x0" in text
    assert "x1" in text
    assert "1.2840" in text
    with pytest.raises(ValidationError, match="feature_names has 1 elements"):
        summary_relativities(result, feature_names=["only_intercept"])
