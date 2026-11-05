"""``fit(robust=...)`` computes robust covariances while fitting (``specs/out-of-core-fit.md`` §4.3, §8).

Every route gives, bit for bit, the sandwich that ``store_design_matrix=True`` computes
on demand: ordinary IRLS and coordinate descent in their final pass, centred
standardisation, smooth fits and Negative Binomial θ estimation from the rows they were
given.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rustystats as rs
from _golden_cases import CASES, GoldenCase
from rustystats.exceptions import ValidationError

HC_TYPES = ["HC0", "HC1", "HC2", "HC3"]


def _has_covariance(case: GoldenCase) -> bool:
    _glm, fit, _holdout = case.build()
    return fit.get("compute_covariance", True)


ROBUST_CASES = [case for case in CASES if _has_covariance(case)]


def _bits(values: Any) -> list[str]:
    return [float(v).hex() for v in np.asarray(values, dtype=np.float64).ravel()]


@pytest.mark.parametrize("case", ROBUST_CASES, ids=[case.name for case in ROBUST_CASES])
def test_fit_time_robust_equals_stored_design_bitwise(case: GoldenCase) -> None:
    glm_kwargs, fit_kwargs, _holdout = case.build()
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if k != "store_design_matrix"}
    stored = rs.glm_dict(**glm_kwargs).fit(store_design_matrix=True, **fit_kwargs)
    glm_kwargs, _fit, _holdout = case.build()
    fit_time = rs.glm_dict(**glm_kwargs).fit(robust=HC_TYPES, **fit_kwargs)
    for hc_type in HC_TYPES:
        try:
            expected = stored.bse_robust(hc_type)
        except Exception as exc:  # the case's model has no robust covariance today
            with pytest.raises(type(exc)):
                fit_time.bse_robust(hc_type)
            continue
        assert _bits(fit_time.bse_robust(hc_type)) == _bits(expected), hc_type
        assert _bits(fit_time._result.cov_robust(hc_type)) == _bits(
            stored._result.cov_robust(hc_type)
        ), hc_type


def _simple_model() -> Any:
    case = next(case for case in CASES if case.name == "robust_hc_all")
    glm_kwargs, _fit, _holdout = case.build()
    return rs.glm_dict(**glm_kwargs)


def test_types_not_listed_still_need_a_stored_design() -> None:
    result = _simple_model().fit(robust=["hc1"])
    assert np.all(np.isfinite(result.bse_robust("HC1")))
    with pytest.raises(ValueError, match="Design matrix not stored"):
        result.bse_robust("HC3")


@pytest.mark.parametrize(
    ("robust", "fit", "message"),
    [
        (["HC4"], {}, "Unknown robust covariance type"),
        ("HC1", {}, "takes a list"),
        (["HC0"], {"compute_covariance": False}, "needs the covariance matrix"),
    ],
    ids=["unknown", "bare_string", "without_covariance"],
)
def test_robust_option_refusals(robust: Any, fit: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _simple_model().fit(robust=robust, **fit)
