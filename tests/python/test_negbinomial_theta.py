"""RS-ACT-010: Negative Binomial theta contract.

Previously ``family="negbinomial"`` silently fell back to ``theta=1.0`` and the
profile-likelihood estimator (``fit_negbinomial_py``) had zero Python callers.
The contract now is:

* ``theta=<number>``  -> fixed theta (every fit path).
* ``theta="estimate"`` -> profile MLE (explicit opt-in, plain GLM only).
* ``theta=None`` (unspecified) -> rejected, so NB never silently chooses a
  dispersion policy.

Estimation must respect offset/weights and record honest metadata.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs


def _nb_frame(n: int = 3000, true_theta: float = 2.0, seed: int = 0, with_exposure: bool = False):
    """Negative-Binomial counts via a Gamma-Poisson mixture (Var = mu + mu^2/theta)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    mu = np.exp(0.5 + 0.4 * x)
    lam = rng.gamma(shape=true_theta, scale=mu / true_theta)
    y = rng.poisson(lam).astype(float)
    cols: dict[str, np.ndarray] = {"y": y, "x": x}
    if with_exposure:
        cols["Exposure"] = rng.uniform(0.5, 2.0, n)
    return pl.DataFrame(cols)


def _fit(data, terms=None, **fit_kwargs):
    terms = terms or {"x": {"type": "linear"}}
    glm_kwargs = {}
    for key in ("theta", "exposure"):
        if key in fit_kwargs:
            glm_kwargs[key] = fit_kwargs.pop(key)
    return rs.glm_dict(
        response="y", terms=terms, data=data, family="negbinomial", **glm_kwargs
    ).fit(**fit_kwargs)


class TestNegBinomialThetaContract:
    def test_unspecified_theta_requires_explicit_policy(self):
        """010.1: no theta raises instead of silently choosing 1.0 or estimation."""
        data = _nb_frame(true_theta=2.0)
        with pytest.raises(rs.ValidationError, match=r"(?i)theta"):
            _fit(data)

    def test_estimate_keyword_records_metadata(self):
        """010.2: theta='estimate' records estimated flag, value, init, iterations, tol."""
        data = _nb_frame(true_theta=2.0)
        result = _fit(data, theta="estimate")
        meta = result.theta_metadata
        assert meta["estimated"] is True
        assert meta["theta"] == pytest.approx(result.theta)
        assert meta["theta_iterations"] >= 1
        assert isinstance(meta["theta_converged"], bool)
        assert meta["init_theta"] > 0
        assert meta["theta_tol"] > 0
        assert meta["glm_tol"] == pytest.approx(1e-8)

    def test_estimate_records_requested_glm_tolerance(self):
        """010.2: the NB estimator records and uses the requested GLM tolerance."""
        data = _nb_frame(n=1200, true_theta=2.0)
        result = _fit(data, theta="estimate", tol=1e-9)
        assert result.theta_metadata["glm_tol"] == pytest.approx(1e-9)

    def test_numeric_theta_is_fixed(self):
        """010.3: numeric theta is recorded as fixed and shown in the family string."""
        data = _nb_frame(true_theta=2.0)
        result = _fit(data, theta=1.5)
        assert result.theta == pytest.approx(1.5)
        assert result.theta_metadata["estimated"] is False
        assert result.family == "NegativeBinomial(theta=1.5000)"

    def test_estimate_and_fixed_metadata_share_one_schema(self):
        """RS-ACT-010: both theta_metadata dicts carry the same keys (incl.
        fallback_reason and glm_tol) so consumers can rely on the schema."""
        data = _nb_frame(n=1200, true_theta=2.0)
        estimated = _fit(data, theta="estimate").theta_metadata
        fixed = _fit(data, theta=1.5).theta_metadata
        assert set(estimated) == set(fixed)
        assert "fallback_reason" in estimated
        assert "glm_tol" in fixed
        assert estimated["fallback_reason"] is None

    def test_theta_metadata_survives_serialization(self):
        """010.2/010.3: serialized NB models keep theta provenance."""
        data = _nb_frame(n=1200, true_theta=2.0)

        estimated = _fit(data, theta="estimate")
        loaded_estimated = rs.GLMModel.from_bytes(estimated.to_bytes())
        assert loaded_estimated.theta == pytest.approx(estimated.theta)
        assert loaded_estimated.theta_metadata == estimated.theta_metadata
        assert loaded_estimated.family == estimated.family

        fixed = _fit(data, theta=1.5)
        loaded_fixed = rs.GLMModel.from_bytes(fixed.to_bytes())
        assert loaded_fixed.theta == pytest.approx(1.5)
        assert loaded_fixed.theta_metadata == fixed.theta_metadata
        assert loaded_fixed.family == "NegativeBinomial(theta=1.5000)"

    def test_numeric_theta_matches_statsmodels_coefficients(self):
        """010.3: at a fixed theta, coefficients match statsmodels NB GLM."""
        sm = pytest.importorskip("statsmodels.api")
        data = _nb_frame(true_theta=2.0)
        result = _fit(data, theta=2.0)

        y = data["y"].to_numpy()
        X = sm.add_constant(data["x"].to_numpy())
        # statsmodels NB2 variance = mu + alpha*mu^2, so alpha = 1/theta.
        sm_res = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=0.5)).fit()
        np.testing.assert_allclose(result.params, sm_res.params, atol=0.02, rtol=0.01)

    def test_estimated_theta_in_statsmodels_ballpark(self):
        """010.4: estimated theta cross-checks against statsmodels' NB MLE."""
        sm = pytest.importorskip("statsmodels.api")
        data = _nb_frame(true_theta=2.0)
        result = _fit(data, theta="estimate")

        y = data["y"].to_numpy()
        X = sm.add_constant(data["x"].to_numpy())
        sm_nb = sm.NegativeBinomial(y, X).fit(disp=0)
        sm_theta = 1.0 / sm_nb.params[-1]  # last param is the dispersion alpha
        # NB theta estimators vary between methods; require the same ballpark.
        assert 0.5 * sm_theta < result.theta < 2.0 * sm_theta

    def test_offset_respected_in_estimation(self):
        """010.4: exposure offset is threaded into the estimator."""
        data = _nb_frame(true_theta=2.0, with_exposure=True)
        result = _fit(data, theta="estimate", exposure="Exposure")
        assert result.theta is not None
        assert result.theta_metadata["estimated"] is True

    def test_weights_respected_in_estimation(self):
        """010.4: non-uniform observation weights are threaded into the estimator.

        Spec 010.4 explicitly says "Offset and weights are respected in theta
        estimation". The matching offset case is already covered above; this
        pins the weights branch. We assert the fit runs without error and
        records ``estimated=True``, and that the estimated theta differs from
        the unweighted estimate (so the weights aren't silently ignored).
        """
        data = _nb_frame(true_theta=2.0)
        rng = np.random.default_rng(99)
        weights = rng.uniform(0.2, 5.0, len(data))

        weighted = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="negbinomial",
            theta="estimate",
            weights=weights,
        ).fit()
        unweighted = _fit(data, theta="estimate")

        assert weighted.theta is not None
        assert np.isfinite(weighted.theta)
        assert weighted.theta_metadata["estimated"] is True
        # Different weights -> different estimated theta. If they came out
        # bit-identical the weights vector would not have reached the
        # profile-MLE inner loop (the regression we're guarding against).
        assert weighted.theta != pytest.approx(unweighted.theta, rel=1e-6, abs=1e-9)

    def test_estimate_with_regularization_raises(self):
        """010.5/010.6: regularized NB + estimate fails closed, asking for explicit theta."""
        data = _nb_frame(true_theta=2.0)
        with pytest.raises(rs.ValidationError, match=r"(?i)theta"):
            _fit(data, theta="estimate", cv=3, regularization="ridge", n_alphas=3, verbose=False)

    def test_estimate_with_smooth_raises(self):
        """010.5: smooth NB + estimate fails closed (smooth solver has no theta loop)."""
        data = _nb_frame(true_theta=2.0)
        with pytest.raises(rs.ValidationError, match=r"(?i)theta"):
            _fit(data, terms={"x": {"type": "bs", "k": 6}}, theta="estimate")

    def test_estimate_with_sign_constraint_raises(self):
        """010.5: sign-constrained NB + estimate fails closed, naming the
        constraint and the ``theta="estimate"`` request.

        The other two unsupported-combo branches (smooth, regularized) already
        have coverage above; the sign-constrained branch is exercised here.
        A linear term with ``"monotonicity": "increasing"`` is rendered as
        ``pos(x)`` in the feature names, which ``_get_constraint_indices``
        recognises as a non-negative coefficient constraint.
        """
        data = _nb_frame(true_theta=2.0)
        with pytest.raises(rs.ValidationError) as excinfo:
            rs.glm_dict(
                response="y",
                terms={"x": {"type": "linear", "monotonicity": "increasing"}},
                data=data,
                family="negbinomial",
                theta="estimate",
            ).fit()
        msg = str(excinfo.value)
        # The error must name the constraint type so the user knows why
        # estimation is refused.
        assert "sign-constrained" in msg
        # And it must point at the theta knob that needs to be fixed.
        assert "theta" in msg

    def test_regularized_negbinomial_with_fixed_theta_works(self):
        """010.6: an explicit numeric theta permits regularized NB."""
        data = _nb_frame(true_theta=2.0)
        result = _fit(data, theta=1.5, cv=3, regularization="ridge", n_alphas=3, verbose=False)
        assert result.cv_deviance is not None
        assert "theta=1.5000" in result.family

    def test_regularization_module_docstring_records_nb_theta_policy(self):
        """010 (review): the regularization-path module docstring must record
        the fixed-theta-only contract and point at the caller in formula.py.

        Pins the doc against a silent drop of the policy note — if the module
        docstring stops mentioning the contract, this test catches it.
        """
        import rustystats.regularization_path as rp

        doc = rp.__doc__ or ""
        assert "RS-ACT-010" in doc, "regularization_path module docstring lost the RS-ACT-010 tag"
        assert "fixed" in doc.lower() and "theta" in doc.lower(), (
            "regularization_path module docstring lost the fixed-theta policy note"
        )
        assert "_resolve_negbinomial_theta" in doc, (
            "regularization_path docstring must point at the policy resolver"
        )
