"""
IRLS Working Response and Weights
=================================

For an arbitrary linear predictor η (not a fitted GLM's η), compute the IRLS
working response z and combined working weight w. Used by boosting loops that
iterate on the link scale — for example, distilling a layered tree ensemble
into a GLM by chaining residuals through the link function.

Formula
-------
    μ = g⁻¹(η + offset)
    z = η + (y − μ) · g'(μ)
    w = prior_weight × IRLS_weight

where the IRLS_weight is either the Fisher information weight
``1 / (V(μ) · g'(μ)²)`` or, for log-link Tweedie with 1 < p < 2, the true
Hessian weight ``μ^(2 − p)``. Gamma + log uses the Fisher form
(``w = 1``) to match statsmodels.

``η`` is the linear predictor *excluding* the offset (e.g. the running
boosted prediction on the link scale). The offset is added internally
to form μ.

Family / link strings follow the rest of rustystats:

* families: ``"poisson"``, ``"gamma"``, ``"gaussian"``, ``"binomial"``,
  ``"tweedie"``, ``"quasipoisson"``, ``"quasibinomial"``, ``"negbinomial"``.
  Embedded forms ``"tweedie(p=1.5)"`` and ``"negativebinomial(theta=1.2)"``
  are also accepted.
* link defaults to the family's canonical link when omitted or ``"default"``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rustystats._rustystats import working_response_weights_py as _wrw_rust
from rustystats.validation import validate_residual_inputs

__all__ = ["working_response_weights"]


def working_response_weights(
    y: npt.ArrayLike,
    eta: npt.ArrayLike,
    family: str,
    link: str | None = "default",
    *,
    offset: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    allow_extended_tweedie: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute IRLS working response z and combined working weight w.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed response.
    eta : array-like, shape (n,)
        Current linear predictor *excluding* offset.
    family : str
        One of ``"poisson"``, ``"gamma"``, ``"gaussian"``, ``"binomial"``,
        ``"tweedie"``, ``"quasipoisson"``, ``"quasibinomial"``,
        ``"negbinomial"``. Embedded parameters are accepted, e.g.
        ``"tweedie(p=1.5)"`` or ``"negativebinomial(theta=1.0)"``.
    link : str, optional
        Link function. One of ``"identity"``, ``"log"``, ``"logit"``. The
        literal string ``"default"`` (or ``None``) resolves to the family's
        canonical link.
    offset : array-like, shape (n,), optional
        Offset on the linear predictor scale (e.g. ``log(exposure)``). Added
        to ``eta`` when computing μ. Defaults to zeros.
    weights : array-like, shape (n,), optional
        Non-negative prior weights. Defaults to ones.
    var_power : float, default 1.5
        Tweedie variance power. Ignored for non-Tweedie families.
    theta : float, default 1.0
        Negative Binomial dispersion. Ignored for non-NB families.
    allow_extended_tweedie : bool, default False
        Permit Tweedie powers outside the default ``1 < p < 2`` interior
        (i.e. ``p <= 0``, ``p == 1``, ``p == 2`` (Gamma), ``p > 2``). The
        per-regime response support rules are enforced regardless.

    Returns
    -------
    z : np.ndarray of float64, shape (n,)
        Working response: ``z = η + (y − μ) · g'(μ)``.
    w : np.ndarray of float64, shape (n,)
        Combined working weight: ``w = prior_weight × IRLS_weight``.

    Notes
    -----
    For log-link Tweedie with 1 < p < 2 this uses the true Hessian weight
    ``μ^(2 − p)`` rather than the Fisher form, matching rustystats's IRLS
    solver. For Gamma + log the Fisher weight (``w = 1``) is used to match
    statsmodels conventions for covariance estimation.

    The per-row computation has no cross-row state, so the output is
    deterministic regardless of how the underlying parallel implementation
    schedules work.

    Examples
    --------
    >>> import numpy as np
    >>> from rustystats import working_response_weights
    >>> y = np.array([0.0, 1.0, 2.0])
    >>> eta = np.zeros(3)  # μ = exp(0) = 1
    >>> z, w = working_response_weights(y, eta, family="poisson")
    >>> bool(np.allclose(z, [-1.0, 0.0, 1.0]))
    True
    >>> bool(np.allclose(w, [1.0, 1.0, 1.0]))
    True
    """
    # Resolve "default" / None to None so the Rust binding picks the canonical link.
    link_resolved = None if link in (None, "default") else link

    y_arr, eta_arr, weights_arr, offset_arr = validate_residual_inputs(
        y,
        eta,
        family,
        weights=weights,
        offset=offset,
    )

    return _wrw_rust(
        y_arr,
        eta_arr,
        family,
        link_resolved,
        var_power,
        theta,
        offset_arr,
        weights_arr,
        allow_extended_tweedie,
    )
