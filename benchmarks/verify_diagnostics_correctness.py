"""
Comprehensive verification harness for the post-OPT-2..OPT-20 diagnostics
pipeline.

Runs `result.diagnostics(...)` across multiple distinct scenarios (Poisson,
Gaussian, Binomial, Gamma, edge cases, target encoding, splines), then asserts
structural completeness, value sanity, cross-validation against statsmodels
where reasonable, numerical equivalence between singular and batch Rust
helpers, JSON round-tripping, and model serialization round-tripping.

Run with:
    uv run python benchmarks/verify_diagnostics_correctness.py
"""

from __future__ import annotations

import json
import math
import sys
import traceback
import warnings as _warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import rustystats as rs
from rustystats._rustystats import (
    compute_ae_by_decile_py as _rust_ae_by_decile,
)
from rustystats._rustystats import (
    compute_ae_categorical_batch_py as _rust_ae_categorical_batch,
)
from rustystats._rustystats import (
    compute_ae_categorical_py as _rust_ae_categorical,
)
from rustystats._rustystats import (
    compute_ae_continuous_batch_py as _rust_ae_continuous_batch,
)
from rustystats._rustystats import (
    compute_ae_continuous_py as _rust_ae_continuous,
)
from rustystats._rustystats import (
    compute_factor_deviance_batch_from_codes_py as _rust_factor_deviance_batch_from_codes,
)
from rustystats._rustystats import (
    compute_factor_deviance_batch_py as _rust_factor_deviance_batch,
)
from rustystats._rustystats import (
    compute_factor_deviance_py as _rust_factor_deviance,
)
from rustystats._rustystats import (
    compute_factor_significance_batch_py as _rust_factor_significance_batch,
)
from rustystats._rustystats import (
    partial_dependence_categorical_batch_py as _rust_partial_dependence_categorical_batch,
)
from rustystats._rustystats import (
    stack_columns_horizontal_py as _rust_stack_columns,
)

try:
    import statsmodels.api as sm
    import statsmodels.genmod.families as sm_fam

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False


# =============================================================================
# Result accumulator
# =============================================================================


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


class CheckCollector:
    """Collects pass/fail results without aborting on first fail."""

    def __init__(self, scenario_label: str):
        self.scenario = scenario_label
        self.results: list[CheckResult] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        msg = "" if condition else detail
        self.results.append(CheckResult(name=name, passed=bool(condition), message=msg))
        return bool(condition)

    def fail(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, passed=False, message=detail))

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def report(self) -> str:
        if self.n_failed == 0:
            return f"Scenario {self.scenario}: {self.n_passed} invariants checked, all passed."
        lines = [f"Scenario {self.scenario}: {self.n_failed}/{self.n_total} FAILED"]
        for r in self.results:
            if not r.passed:
                lines.append(f"  FAIL [{r.name}]: {r.message}")
        return "\n".join(lines)


# =============================================================================
# Generic diagnostic invariant checks (shared by all scenarios)
# =============================================================================


def _is_finite_float(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_top_level_completeness(
    diag,
    c: CheckCollector,
    *,
    expect_lift: bool = True,
    expect_partial_dep: bool = True,
    expect_factor_dev: bool = True,
    expect_vif: bool = True,
    expect_coef_summary: bool = True,
) -> None:
    """All required top-level fields present and non-None.

    `factor_deviance` is intentionally only populated when at least one
    categorical factor is fitted (see api.py:506). Callers must pass
    expect_factor_dev=False when there are no categorical factors.
    """
    c.check("top.model_summary present", diag.model_summary is not None, "model_summary is None")
    c.check("top.train_test present", diag.train_test is not None, "train_test is None")
    c.check(
        "top.calibration present",
        diag.calibration is not None and isinstance(diag.calibration, dict),
        "calibration is None",
    )
    c.check(
        "top.residual_summary present",
        diag.residual_summary is not None and isinstance(diag.residual_summary, dict),
        "residual_summary is None",
    )
    c.check(
        "top.factors present",
        diag.factors is not None and isinstance(diag.factors, list),
        "factors is None or not a list",
    )
    if expect_vif:
        c.check(
            "top.vif present", diag.vif is not None and isinstance(diag.vif, list), "vif is None"
        )
    if expect_coef_summary:
        c.check(
            "top.coefficient_summary present",
            diag.coefficient_summary is not None and len(diag.coefficient_summary) > 0,
            "coefficient_summary is None or empty",
        )
    if expect_factor_dev:
        c.check(
            "top.factor_deviance present",
            diag.factor_deviance is not None and isinstance(diag.factor_deviance, list),
            "factor_deviance is None",
        )
    else:
        # Sanity: when we don't expect it, also confirm it is None (the
        # documented contract: no categorical factors -> no factor_deviance).
        c.check(
            "top.factor_deviance None when no cat factors fitted",
            diag.factor_deviance is None,
            f"factor_deviance unexpectedly populated: {diag.factor_deviance}",
        )
    if expect_lift:
        c.check("top.lift_chart present", diag.lift_chart is not None, "lift_chart is None")
    if expect_partial_dep:
        c.check(
            "top.partial_dependence present",
            diag.partial_dependence is not None and isinstance(diag.partial_dependence, list),
            "partial_dependence is None",
        )


def check_model_summary(diag, c: CheckCollector) -> None:
    """model_summary required keys + sane scalar values."""
    ms = diag.model_summary or {}
    required = [
        "formula",
        "family",
        "link",
        "n_obs",
        "n_params",
        "df_resid",
        "converged",
        "iterations",
        "scale",
        "scale_pearson",
        "null_deviance",
    ]
    for k in required:
        c.check(
            f"model_summary.{k} present",
            k in ms,
            f"missing key '{k}' (have: {list(ms.keys())[:8]}...)",
        )
    if "n_obs" in ms:
        c.check("model_summary.n_obs > 0", ms["n_obs"] > 0, f"n_obs={ms['n_obs']}")
    if "n_params" in ms:
        c.check("model_summary.n_params > 0", ms["n_params"] > 0, f"n_params={ms['n_params']}")
    if "df_resid" in ms:
        c.check("model_summary.df_resid >= 0", ms["df_resid"] >= 0, f"df_resid={ms['df_resid']}")


def check_train_diag(diag, c: CheckCollector, result=None) -> None:
    """train_test.train field structure + value sanity."""
    tt = diag.train_test
    if tt is None:
        return
    train = tt.train
    if train is None:
        c.fail("train_test.train", "is None")
        return
    # FIX-J: train_test.train.deviance is the textbook GLM deviance
    # (sum of unit deviances), so it should agree with result.deviance modulo
    # the 2-decimal rounding applied by DiagnosticsComputer.
    if result is not None:
        rs_dev = float(result.deviance)
        tt_dev = float(train.deviance)
        if math.isfinite(rs_dev) and math.isfinite(tt_dev):
            c.check(
                "train_test.train.deviance ~ result.deviance (FIX-J)",
                math.isclose(rs_dev, tt_dev, rel_tol=1e-3, abs_tol=max(0.01, 0.005 * abs(rs_dev))),
                f"result.deviance={rs_dev:.6f} train.deviance={tt_dev:.6f}",
            )
    required = [
        "n_obs",
        "deviance",
        "log_likelihood",
        "aic",
        "bic",
        "gini",
        "auc",
        "ae_ratio",
        "ae_by_decile",
        "factor_diagnostics",
    ]
    for attr in required:
        c.check(f"train.{attr} present", hasattr(train, attr), f"missing attr '{attr}'")
    c.check("train.n_obs > 0", train.n_obs > 0, f"n_obs={train.n_obs}")
    c.check("train.aic > 0", train.aic > 0, f"aic={train.aic}")
    c.check("train.bic > 0", train.bic > 0, f"bic={train.bic}")
    # BIC >= AIC for n_obs > exp(2) ≈ 7.4 (penalty is log(n)*p vs 2*p)
    if train.n_obs > 8:
        c.check(
            "train.bic >= train.aic (n>8)",
            train.bic >= train.aic,
            f"aic={train.aic} bic={train.bic}",
        )
    c.check("train.deviance >= 0", train.deviance >= -1e-9, f"deviance={train.deviance}")
    c.check(
        "train.log_likelihood finite",
        _is_finite_float(train.log_likelihood),
        f"log_likelihood={train.log_likelihood}",
    )
    if train.gini is not None:
        c.check("train.gini in [-1, 1]", -1.0 <= train.gini <= 1.0, f"gini={train.gini}")
    if train.auc is not None and not math.isnan(train.auc):
        # AUC can dip below 0.5 for poorly calibrated models, so allow [0, 1].
        c.check("train.auc in [0, 1]", 0.0 <= train.auc <= 1.0, f"auc={train.auc}")
    if train.ae_ratio is not None:
        c.check(
            "train.ae_ratio > 0 and finite",
            train.ae_ratio > 0 and _is_finite_float(train.ae_ratio),
            f"ae_ratio={train.ae_ratio}",
        )
    c.check(
        "train.ae_by_decile == 10", len(train.ae_by_decile) == 10, f"len={len(train.ae_by_decile)}"
    )
    for i, d in enumerate(train.ae_by_decile):
        if d.ae_ratio is not None:
            c.check(
                f"train.decile[{i}].ae_ratio > 0",
                d.ae_ratio > 0 or d.predicted == 0.0,
                f"ae_ratio={d.ae_ratio}",
            )


def check_factors(
    diag,
    c: CheckCollector,
    expected_in_model: set[str],
    expected_unfitted: set[str],
    has_design_matrix: bool,
) -> None:
    """Each factor entry has the right structure."""
    factors = diag.factors or []
    factor_names = {f.name for f in factors}
    for name in expected_in_model | expected_unfitted:
        c.check(
            f"factor '{name}' present",
            name in factor_names,
            f"missing factor '{name}' from list (have: {sorted(factor_names)})",
        )
    for f in factors:
        c.check(
            f"factor[{f.name}].factor_type set",
            f.factor_type in ("categorical", "continuous"),
            f"factor_type={f.factor_type!r}",
        )
        c.check(
            f"factor[{f.name}].in_model is bool",
            isinstance(f.in_model, bool),
            f"in_model={f.in_model!r}",
        )
        c.check(
            f"factor[{f.name}].actual_vs_expected list",
            isinstance(f.actual_vs_expected, list),
            "ae list missing",
        )
        # Bin-level checks
        for i, b in enumerate(f.actual_vs_expected):
            if not c.check(
                f"factor[{f.name}].ae[{i}].bin non-empty",
                isinstance(b.bin, str) and len(b.bin) > 0,
                f"bin={b.bin!r}",
            ):
                continue
            c.check(f"factor[{f.name}].ae[{i}].n > 0", b.n > 0, f"n={b.n}")
            c.check(
                f"factor[{f.name}].ae[{i}].exposure >= 0", b.exposure >= 0, f"exposure={b.exposure}"
            )
            if b.ae_ratio is not None and not math.isnan(b.ae_ratio):
                c.check(
                    f"factor[{f.name}].ae[{i}].ae_ratio finite",
                    _is_finite_float(b.ae_ratio),
                    f"ae_ratio={b.ae_ratio}",
                )
            c.check(
                f"factor[{f.name}].ae[{i}].ae_ci length 2",
                isinstance(b.ae_ci, list) and len(b.ae_ci) == 2,
                f"ae_ci={b.ae_ci}",
            )
        c.check(
            f"factor[{f.name}].residual_pattern present",
            f.residual_pattern is not None,
            "residual_pattern is None",
        )
        if f.in_model:
            # coefficients can be None when the factor is folded into a
            # transform whose coefficients sit elsewhere (e.g. TE), so
            # only check the factor is marked correctly.
            pass
        else:
            # Unfitted factor + design matrix => score test should be present.
            if has_design_matrix:
                if not c.check(
                    f"factor[{f.name}].score_test present (unfitted)",
                    f.score_test is not None,
                    "score_test missing for unfitted factor",
                ):
                    continue
                c.check(
                    f"factor[{f.name}].score_test.statistic finite",
                    _is_finite_float(f.score_test.statistic),
                    f"statistic={f.score_test.statistic}",
                )
                c.check(
                    f"factor[{f.name}].score_test.pvalue in [0,1]",
                    0.0 <= f.score_test.pvalue <= 1.0,
                    f"pvalue={f.score_test.pvalue}",
                )


def check_lift(diag, c: CheckCollector) -> None:
    lift = diag.lift_chart
    if lift is None:
        return
    c.check("lift.deciles == 10", len(lift.deciles) == 10, f"len={len(lift.deciles)}")
    if lift.gini is not None and not math.isnan(lift.gini):
        c.check("lift.gini in [-1, 1]", -1.0 <= lift.gini <= 1.0, f"gini={lift.gini}")


def check_partial_dependence(diag, c: CheckCollector, expected_factors: set[str]) -> None:
    pd_list = diag.partial_dependence
    if pd_list is None:
        return
    pd_names = {p.variable for p in pd_list}
    for name in expected_factors:
        c.check(f"partial_dependence has '{name}'", name in pd_names, f"missing PD for '{name}'")
    for p in pd_list:
        c.check(
            f"PD[{p.variable}].grid_values non-empty", len(p.grid_values) > 0, "empty grid_values"
        )
        c.check(
            f"PD[{p.variable}].predictions non-empty", len(p.predictions) > 0, "empty predictions"
        )
        c.check(
            f"PD[{p.variable}].grid same len as predictions",
            len(p.grid_values) == len(p.predictions),
            f"grid={len(p.grid_values)} pred={len(p.predictions)}",
        )
        c.check(
            f"PD[{p.variable}].shape non-empty",
            isinstance(p.shape, str) and len(p.shape) > 0,
            f"shape={p.shape!r}",
        )
        c.check(
            f"PD[{p.variable}].recommendation non-empty",
            isinstance(p.recommendation, str) and len(p.recommendation) > 0,
            f"recommendation={p.recommendation!r}",
        )


def check_vif(diag, c: CheckCollector) -> None:
    if diag.vif is None:
        return
    for v in diag.vif:
        c.check(
            f"VIF[{v.feature}] finite or capped",
            _is_finite_float(v.vif) or v.vif <= 1000.0,
            f"vif={v.vif}",
        )
        # Non-intercept vif should be >= 1 (capped at 999 in code)
        c.check(
            f"VIF[{v.feature}].vif >= 1.0",
            v.vif >= 0.999,  # allow tiny numerical noise
            f"vif={v.vif}",
        )
        c.check(
            f"VIF[{v.feature}].severity present",
            v.severity in ("none", "moderate", "severe", "expected"),
            f"severity={v.severity!r}",
        )


def check_coefficient_summary(diag, c: CheckCollector) -> None:
    if diag.coefficient_summary is None:
        return
    for cs in diag.coefficient_summary:
        c.check(
            f"coef[{cs.feature}].estimate finite",
            _is_finite_float(cs.estimate),
            f"estimate={cs.estimate}",
        )
        c.check(
            f"coef[{cs.feature}].std_error finite or NaN",
            math.isnan(cs.std_error) or _is_finite_float(cs.std_error),
            f"std_error={cs.std_error}",
        )
        c.check(
            f"coef[{cs.feature}].p_value in [0,1] or NaN",
            math.isnan(cs.p_value) or 0.0 <= cs.p_value <= 1.0,
            f"p_value={cs.p_value}",
        )
        c.check(
            f"coef[{cs.feature}].significant is bool",
            isinstance(cs.significant, bool),
            f"significant={cs.significant!r}",
        )


def check_factor_deviance(diag, c: CheckCollector) -> None:
    if diag.factor_deviance is None:
        return
    for fd in diag.factor_deviance:
        if fd.total_deviance is None:
            continue
        # Sum of per-level deviance should approximate total deviance
        # contribution. Allow modest tolerance for rounding (we round to 2 dp).
        sum_dev = sum(level.deviance for level in fd.levels if level.deviance is not None)
        # Total deviance is rounded to 2 dp in the dataclass; allow 1.0
        # absolute tolerance because per-level deviances are also rounded.
        c.check(
            f"factor_dev[{fd.factor}].sum_levels ~ total",
            math.isclose(
                sum_dev,
                fd.total_deviance,
                rel_tol=0.05,
                abs_tol=max(1.0, 0.01 * abs(fd.total_deviance)),
            ),
            f"sum_levels={sum_dev:.2f} total={fd.total_deviance:.2f}",
        )


def check_calibration(diag, c: CheckCollector) -> None:
    cal = diag.calibration or {}
    if "deciles" in cal:
        deciles = cal["deciles"]
        # Cumulative actual vs predicted should track.
        cum_actual = 0.0
        cum_pred = 0.0
        for d in deciles:
            actual_sum = d.get("actual_sum", 0.0) or 0.0
            pred_sum = d.get("predicted_sum", 0.0) or 0.0
            cum_actual += actual_sum
            cum_pred += pred_sum
        if cum_pred > 0:
            ratio = cum_actual / cum_pred
            c.check(
                "calibration cumulative actual ~ predicted (ratio in [0.5, 2.0])",
                0.5 <= ratio <= 2.0,
                f"cum_actual={cum_actual:.2f} cum_pred={cum_pred:.2f} ratio={ratio:.3f}",
            )


# =============================================================================
# Cross-validation against statsmodels
# =============================================================================


def cross_check_statsmodels(
    family_name: str,
    link_name: str,
    diag,
    rs_result,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    c: CheckCollector,
) -> None:
    """Compare key scalar diagnostics with statsmodels.

    Notes:
    - We compare against `result.deviance` and `result.params` directly (the
      raw fitted values). As of FIX-J, `diag.train_test.train.deviance` also
      reports the textbook GLM deviance (sum of unit deviances) and so should
      agree with `result.deviance` and statsmodels to within rounding. The
      `loss` field on `train_test.train` continues to expose the family-specific
      loss (Binomial NLL, Gaussian MSE, etc.) used by GBMs.
    - `diag.coefficient_summary` is reordered (e.g. by p-value), so we
      rebuild a name -> estimate dict before comparing to statsmodels.
    """
    if not HAS_STATSMODELS:
        return
    family_map = {
        "gaussian": sm_fam.Gaussian,
        "poisson": sm_fam.Poisson,
        "binomial": sm_fam.Binomial,
        "gamma": sm_fam.Gamma,
    }
    link_map = {
        "identity": sm_fam.links.Identity,
        "log": sm_fam.links.Log,
        "logit": sm_fam.links.Logit,
        "inverse": sm_fam.links.InversePower,
    }
    fam_cls = family_map.get(family_name.lower())
    link_cls = link_map.get(link_name.lower())
    if fam_cls is None or link_cls is None:
        return
    try:
        fam_instance = fam_cls(link=link_cls())
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            sm_model = sm.GLM(y, X, family=fam_instance, freq_weights=weights)
            sm_res = sm_model.fit(maxiter=200)
    except Exception as e:
        c.fail("statsmodels.fit", f"failed: {e}")
        return

    # Compare raw fitted coefficients (not the reordered coefficient_summary).
    rs_params_arr = np.asarray(rs_result.params, dtype=np.float64)
    sm_params_arr = np.asarray(sm_res.params, dtype=np.float64)
    if len(rs_params_arr) == len(sm_params_arr):
        max_diff = float(np.max(np.abs(rs_params_arr - sm_params_arr)))
        c.check(
            "sm.coefficients match (max abs diff <= 1e-3)",
            max_diff <= 1e-3,
            f"max abs diff = {max_diff:.6e}; "
            f"rs={rs_params_arr[:5].tolist()}, sm={sm_params_arr[:5].tolist()}",
        )
    else:
        c.fail("sm.coefficients length", f"rs has {len(rs_params_arr)} sm has {len(sm_params_arr)}")

    # Standard errors: bse vs sm bse, allow 5% tolerance per the spec.
    try:
        rs_bse = np.asarray(rs_result.bse(), dtype=np.float64)
        sm_bse = np.asarray(sm_res.bse, dtype=np.float64)
        if len(rs_bse) == len(sm_bse):
            valid = np.isfinite(rs_bse) & np.isfinite(sm_bse) & (sm_bse > 0)
            if np.any(valid):
                rel_diff = np.max(np.abs(rs_bse[valid] - sm_bse[valid]) / sm_bse[valid])
                c.check("sm.bse within 5%", rel_diff <= 0.05, f"max rel diff = {rel_diff:.4%}")
    except Exception as e:
        c.fail("sm.bse compare", f"failed: {e}")

    # Compare *raw* deviance from result (not the dataset_metrics loss).
    sm_dev = float(sm_res.deviance)
    rs_dev = float(rs_result.deviance)
    if math.isfinite(sm_dev) and math.isfinite(rs_dev):
        c.check(
            "sm.deviance within 1% (vs result.deviance)",
            math.isclose(
                rs_dev,
                sm_dev,
                rel_tol=0.01,
                abs_tol=max(0.5, 0.005 * max(abs(rs_dev), abs(sm_dev))),
            ),
            f"rs={rs_dev:.4f} sm={sm_dev:.4f}",
        )

    # Null deviance from model_summary
    sm_null_dev = float(sm_res.null_deviance)
    rs_null_dev = diag.model_summary.get("null_deviance") if diag.model_summary else None
    if rs_null_dev is not None and math.isfinite(sm_null_dev):
        c.check(
            "sm.null_deviance within 1%",
            math.isclose(
                float(rs_null_dev),
                sm_null_dev,
                rel_tol=0.01,
                abs_tol=max(0.5, 0.005 * max(abs(float(rs_null_dev)), abs(sm_null_dev))),
            ),
            f"rs={rs_null_dev} sm={sm_null_dev:.4f}",
        )

    # AIC: statsmodels and rustystats should match. Note that the
    # train.aic comes from compute_fit_statistics which (for binomial &
    # Gaussian) uses the family loss above and so will differ; compare
    # against statsmodels by computing AIC from log-likelihood with our
    # formula AIC = -2 * llf + 2 * k.
    # Use sm_res.llf which is the proper log-likelihood.
    sm_aic = -2.0 * float(sm_res.llf) + 2.0 * len(sm_params_arr)
    # rustystats train.aic = 2 * mean_loss * n + 2 * k for some families;
    # to compare apples-to-apples, also compute AIC from sm.aic if available.
    sm_aic_official = float(sm_res.aic)
    # Accept either matching official sm.aic OR matching the LLF-derived one.
    rs_aic = float(diag.train_test.train.aic) if diag.train_test else None
    if rs_aic is not None and math.isfinite(rs_aic) and math.isfinite(sm_aic_official):
        diff_official = abs(rs_aic - sm_aic_official) / max(1.0, abs(sm_aic_official))
        diff_llf = abs(rs_aic - sm_aic) / max(1.0, abs(sm_aic))
        c.check(
            "sm.aic within 1% (LLF or official)",
            diff_official <= 0.01 or diff_llf <= 0.01,
            f"rs={rs_aic:.4f} sm_official={sm_aic_official:.4f} sm_llf_aic={sm_aic:.4f}",
        )


# =============================================================================
# Singular vs Batch numerical equivalence
# =============================================================================


def check_singular_vs_batch(
    c: CheckCollector,
    scenario: str,
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None,
    cat_values: np.ndarray,
    cont_values: np.ndarray,
    family_name: str,
) -> None:
    """Validate singular Rust functions match batched ones for the same inputs."""

    # ===== Continuous A/E: singular vs batch =====
    sing_cont = _rust_ae_continuous(cont_values, y, mu, exposure, 10, family_name)
    batch_cont_list = [np.ascontiguousarray(cont_values, dtype=np.float64)]
    batch_cont = _rust_ae_continuous_batch(batch_cont_list, y, mu, exposure, 10, family_name)
    c.check(
        f"{scenario} ae_continuous singular vs batch len",
        len(sing_cont) == len(batch_cont[0]),
        f"singular={len(sing_cont)} batch={len(batch_cont[0])}",
    )
    for i, (s, b) in enumerate(zip(sing_cont, batch_cont[0])):
        if s["count"] == 0 and b["count"] == 0:
            continue
        for key in (
            "count",
            "exposure",
            "actual_sum",
            "predicted_sum",
            "actual_expected_ratio",
            "ae_ci_lower",
            "ae_ci_upper",
        ):
            sval = s.get(key, 0.0)
            bval = b.get(key, 0.0)
            if isinstance(sval, int | float) and isinstance(bval, int | float):
                if math.isnan(sval) and math.isnan(bval):
                    continue
                if not math.isclose(float(sval), float(bval), rel_tol=1e-9, abs_tol=1e-9):
                    c.fail(
                        f"{scenario} ae_continuous batch[{i}].{key}",
                        f"singular={sval} batch={bval}",
                    )

    # ===== Categorical A/E: singular vs batch =====
    cat_str = [str(v) for v in cat_values]
    sing_cat = _rust_ae_categorical(cat_str, y, mu, exposure, 1.0, 20, family_name)
    # Build codes for the batch
    unique_levels, inverse = np.unique(cat_str, return_inverse=True)
    codes_arr = np.ascontiguousarray(inverse, dtype=np.uint32)
    # The factor_deviance_batch_from_codes path still takes a 2D matrix; the
    # newer A/E batch path takes a list of 1D arrays.
    codes_matrix = codes_arr.reshape(-1, 1).astype(np.uint32, copy=False)
    batch_cat = _rust_ae_categorical_batch(
        [codes_arr], [list(unique_levels)], y, mu, exposure, 1.0, 20, family_name
    )
    c.check(
        f"{scenario} ae_categorical singular vs batch len",
        len(sing_cat) == len(batch_cat[0]),
        f"singular={len(sing_cat)} batch={len(batch_cat[0])}",
    )
    # Compare bin-level (rare-bucketing should be deterministic)
    sing_by_label = {b["bin_label"]: b for b in sing_cat}
    batch_by_label = {b["bin_label"]: b for b in batch_cat[0]}
    common_labels = set(sing_by_label.keys()) & set(batch_by_label.keys())
    c.check(
        f"{scenario} ae_categorical batch bin_labels match singular",
        sing_by_label.keys() == batch_by_label.keys(),
        f"singular only: {set(sing_by_label.keys()) - set(batch_by_label.keys())} "
        f"batch only: {set(batch_by_label.keys()) - set(sing_by_label.keys())}",
    )
    for label in common_labels:
        s = sing_by_label[label]
        b = batch_by_label[label]
        for key in ("count", "exposure", "actual_sum", "predicted_sum", "actual_expected_ratio"):
            sval = s.get(key, 0.0)
            bval = b.get(key, 0.0)
            if isinstance(sval, int | float) and isinstance(bval, int | float):
                if math.isnan(sval) and math.isnan(bval):
                    continue
                if not math.isclose(float(sval), float(bval), rel_tol=1e-9, abs_tol=1e-9):
                    c.fail(
                        f"{scenario} ae_categorical batch[{label}].{key}",
                        f"singular={sval} batch={bval}",
                    )

    # ===== Factor deviance: singular vs batch (string and codes) =====
    sing_fdev = _rust_factor_deviance("factorA", cat_str, y, mu, family_name, 1.5, 1.0)
    batch_fdev = _rust_factor_deviance_batch(["factorA"], [cat_str], y, mu, family_name, 1.5, 1.0)
    c.check(
        f"{scenario} factor_dev singular vs batch total",
        math.isclose(
            sing_fdev["total_deviance"], batch_fdev[0]["total_deviance"], rel_tol=1e-9, abs_tol=1e-9
        ),
        f"singular={sing_fdev['total_deviance']} batch={batch_fdev[0]['total_deviance']}",
    )
    # Batch from codes
    batch_codes_fdev = _rust_factor_deviance_batch_from_codes(
        ["factorA"],
        codes_matrix,
        [list(unique_levels)],
        y,
        mu,
        family_name,
        1.5,
        1.0,
    )
    c.check(
        f"{scenario} factor_dev codes batch matches singular total",
        math.isclose(
            sing_fdev["total_deviance"],
            batch_codes_fdev[0]["total_deviance"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        f"singular={sing_fdev['total_deviance']} "
        f"codes_batch={batch_codes_fdev[0]['total_deviance']}",
    )
    # Per-level too
    sing_levels = {l["level"]: l for l in sing_fdev["levels"]}
    codes_levels = {l["level"]: l for l in batch_codes_fdev[0]["levels"]}
    for level, sv in sing_levels.items():
        bv = codes_levels.get(level)
        if bv is None:
            c.fail(f"{scenario} factor_dev codes[{level}] missing", "")
            continue
        for key in ("deviance", "count", "actual_sum", "predicted_sum"):
            if not math.isclose(float(sv[key]), float(bv[key]), rel_tol=1e-9, abs_tol=1e-9):
                c.fail(
                    f"{scenario} factor_dev codes[{level}].{key}",
                    f"singular={sv[key]} batch={bv[key]}",
                )

    # ===== ae_by_decile: with vs without sort_idx =====
    # RS-ACT-004: the internal default ranks by predicted rate (mu/exposure) with
    # index tie-breaking, so the externally supplied sort_idx must match that to
    # reproduce the no-sort result. A stable argsort on the rate key replicates
    # the Rust comparator (rate, then original index).
    if exposure is not None:
        safe_exp = np.where(exposure > 0.0, exposure, 1.0)
        rank_key = mu / safe_exp
    else:
        rank_key = mu
    sort_idx = np.argsort(rank_key, kind="stable").astype(np.uintp)
    rust_no_sort = _rust_ae_by_decile(y, mu, exposure, 10, None)
    rust_with_sort = _rust_ae_by_decile(y, mu, exposure, 10, sort_idx)
    c.check(
        f"{scenario} ae_by_decile len(=10)",
        len(rust_no_sort) == 10 and len(rust_with_sort) == 10,
        f"no_sort={len(rust_no_sort)} with_sort={len(rust_with_sort)}",
    )
    for i, (a, b) in enumerate(zip(rust_no_sort, rust_with_sort)):
        for key in ("n", "actual_sum", "predicted_sum", "exposure_sum"):
            if not math.isclose(float(a[key]), float(b[key]), rel_tol=1e-9, abs_tol=1e-9):
                c.fail(
                    f"{scenario} ae_by_decile sort_idx[{i}].{key}",
                    f"no_sort={a[key]} with_sort={b[key]}",
                )

    # ===== Partial dependence categorical batch: vs np.bincount manual =====
    counts_manual = np.bincount(inverse, minlength=len(unique_levels)).astype(np.float64)
    mu_sums_manual = np.bincount(inverse, weights=mu, minlength=len(unique_levels))
    batch_pd = _rust_partial_dependence_categorical_batch([codes_arr], mu, [len(unique_levels)])
    counts_rust, mu_sums_rust = batch_pd[0]
    counts_rust_arr = np.asarray(counts_rust, dtype=np.float64)
    mu_sums_rust_arr = np.asarray(mu_sums_rust, dtype=np.float64)
    c.check(
        f"{scenario} pd_categorical_batch counts match manual",
        np.allclose(counts_manual, counts_rust_arr, rtol=1e-9, atol=1e-9),
        f"max diff = {float(np.max(np.abs(counts_manual - counts_rust_arr)))}",
    )
    c.check(
        f"{scenario} pd_categorical_batch mu_sums match manual",
        np.allclose(mu_sums_manual, mu_sums_rust_arr, rtol=1e-9, atol=1e-9),
        f"max diff = {float(np.max(np.abs(mu_sums_manual - mu_sums_rust_arr)))}",
    )

    # ===== stack_columns_horizontal: vs np.hstack on same columns =====
    a = np.random.rand(len(y), 3).astype(np.float64)
    b = np.random.rand(len(y), 2).astype(np.float64)
    cd = np.random.rand(len(y), 1).astype(np.float64)
    rust_stacked = _rust_stack_columns([a, b, cd])
    py_stacked = np.hstack([a, b, cd])
    c.check(
        f"{scenario} stack_columns_horizontal matches np.hstack",
        np.allclose(rust_stacked, py_stacked, rtol=1e-12, atol=1e-12),
        f"max diff = {float(np.max(np.abs(rust_stacked - py_stacked)))}",
    )


def check_factor_significance_singular_vs_batch(
    c: CheckCollector, scenario: str, result, factor_names: list[str]
) -> None:
    """Compare the singular Python compute_factor_significance against the new
    Rust batch implementation for one or more factors."""
    from rustystats.diagnostics.factors import _FactorDiagnosticsComputer

    bread = result.get_bread_matrix() if hasattr(result, "get_bread_matrix") else None
    if bread is None:
        return  # nothing we can compare
    params = np.asarray(result.params, dtype=np.float64)
    bse_attr = result.bse
    bse = np.asarray(bse_attr() if callable(bse_attr) else bse_attr, dtype=np.float64)
    feature_names = list(result.feature_names)

    # Build a dummy computer (we only need feature_names and the singular
    # method to be callable).
    fdc = _FactorDiagnosticsComputer.__new__(_FactorDiagnosticsComputer)
    fdc.feature_names = feature_names

    # Build per-factor parameter index lists (the same logic the batch uses).
    indices_per_factor: list[list[int]] = []
    valid_factors: list[str] = []
    for name in factor_names:
        idx = [i for i, fn in enumerate(feature_names) if name in fn and fn != "Intercept"]
        if not idx:
            continue
        valid_factors.append(name)
        indices_per_factor.append(idx)

    if not valid_factors:
        return

    # Batch call
    raw_batch = _rust_factor_significance_batch(
        indices_per_factor, params, bse, np.ascontiguousarray(bread, dtype=np.float64)
    )

    # Singular call (the Python helper)
    for fname, raw in zip(valid_factors, raw_batch):
        sing = fdc.compute_factor_significance(fname, result, bread_matrix=bread)
        if raw is None:
            c.check(
                f"{scenario} factor_sig[{fname}] singular None when batch None",
                sing is None,
                f"singular returned {sing}",
            )
            continue
        if sing is None:
            c.fail(
                f"{scenario} factor_sig[{fname}] singular None but batch produced",
                f"batch chi2={raw['chi2']:.4f}",
            )
            continue
        # Both return values rounded to 2/4 dp; allow small numerical noise.
        c.check(
            f"{scenario} factor_sig[{fname}] chi2 matches",
            math.isclose(float(raw["chi2"]), float(sing.chi2), rel_tol=1e-6, abs_tol=1e-2),
            f"singular={sing.chi2} batch={raw['chi2']}",
        )
        c.check(
            f"{scenario} factor_sig[{fname}] pvalue matches",
            math.isclose(float(raw["pvalue"]), float(sing.p), rel_tol=1e-6, abs_tol=1e-3),
            f"singular={sing.p} batch={raw['pvalue']}",
        )


# =============================================================================
# JSON round-trip
# =============================================================================


def check_json_roundtrip(diag, c: CheckCollector) -> None:
    js = diag.to_json()
    c.check("json non-empty", len(js) > 100, f"len={len(js)}")
    # Should be valid JSON
    try:
        parsed = json.loads(js)
    except Exception as e:
        c.fail("json parses", f"{e}")
        return
    # No NaN strings
    c.check(
        "json has no 'NaN' string tokens",
        "NaN" not in js,
        "found 'NaN' literal in serialized output",
    )
    c.check(
        "json has no 'Infinity' string tokens",
        "Infinity" not in js,
        "found 'Infinity' literal in serialized output",
    )
    # Major fields preserved
    c.check(
        "json roundtrip: model_summary present",
        "model_summary" in parsed and parsed["model_summary"],
        "model_summary missing",
    )
    c.check(
        "json roundtrip: train_test present",
        "train_test" in parsed and parsed["train_test"] is not None,
        "train_test missing",
    )
    if diag.train_test and diag.train_test.train:
        roundtripped = parsed["train_test"]["train"]
        # AIC should match within JSON float precision (we round to 4 dp).
        rs_aic = diag.train_test.train.aic
        if roundtripped.get("aic") is not None and math.isfinite(rs_aic):
            c.check(
                "json roundtrip: train.aic preserved",
                math.isclose(roundtripped["aic"], rs_aic, abs_tol=1e-2, rel_tol=1e-3),
                f"orig={rs_aic} roundtripped={roundtripped['aic']}",
            )


# =============================================================================
# Model serialization (predict round-trip)
# =============================================================================


def check_model_serialization(result, data: pl.DataFrame, c: CheckCollector) -> None:
    raw = result.to_bytes()
    c.check("to_bytes non-empty", len(raw) > 100, f"len={len(raw)}")
    loaded = rs.GLMModel.from_bytes(raw)
    pred_orig = np.asarray(result.predict(data), dtype=np.float64)
    pred_loaded = np.asarray(loaded.predict(data), dtype=np.float64)
    c.check(
        "loaded.predict same shape",
        pred_orig.shape == pred_loaded.shape,
        f"orig={pred_orig.shape} loaded={pred_loaded.shape}",
    )
    if pred_orig.shape == pred_loaded.shape:
        max_diff = float(np.max(np.abs(pred_orig - pred_loaded)))
        c.check("loaded.predict max diff < 1e-9", max_diff < 1e-9, f"max diff = {max_diff:.6e}")


# =============================================================================
# Scenario builders
# =============================================================================


def make_poisson_small(seed: int = 11) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 10_000
    age = rng.uniform(18, 70, n).astype(np.float64)
    region = rng.choice(["A", "B", "C", "D"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    veh_power = rng.integers(4, 12, n).astype(np.float64)
    bonus = rng.normal(0.5, 0.2, n).astype(np.float64)
    extra_cont = rng.standard_normal(n).astype(np.float64)
    exposure = rng.uniform(0.2, 1.0, n).astype(np.float64)
    region_eff = np.where(
        region == "A", 0.0, np.where(region == "B", 0.1, np.where(region == "C", -0.1, 0.2))
    )
    eta = -1.5 + 0.01 * (age - 40) + region_eff + 0.05 * (veh_power - 6) + 0.2 * bonus
    mu = np.exp(eta) * exposure
    y = rng.poisson(mu).astype(np.float64)
    df = pl.DataFrame(
        {
            "y": y,
            "exposure": exposure,
            "age": age,
            "region": region,
            "veh_power": veh_power,
            "bonus": bonus,
            "extra_cont": extra_cont,
        }
    )
    return df, {
        "response": "y",
        "offset": "exposure",
        "family": "poisson",
        "terms": {
            "age": {"type": "linear"},
            "region": {"type": "categorical"},
            "veh_power": {"type": "linear"},
            "bonus": {"type": "bs", "df": 4},
        },
    }


def make_poisson_with_te(seed: int = 13) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 50_000
    age = rng.uniform(18, 70, n).astype(np.float64)
    veh_brand = rng.integers(0, 50, n).astype(int)
    region = rng.choice(["A", "B", "C", "D", "E"], size=n)
    bonus = rng.normal(0.5, 0.2, n).astype(np.float64)
    veh_power = rng.integers(4, 12, n).astype(np.float64)
    exposure = rng.uniform(0.2, 1.0, n).astype(np.float64)
    eta = -1.5 + 0.01 * (age - 40) + 0.05 * (veh_power - 6) + 0.005 * veh_brand
    mu = np.exp(eta) * exposure
    y = rng.poisson(mu).astype(np.float64)
    df = pl.DataFrame(
        {
            "y": y,
            "exposure": exposure,
            "age": age,
            "veh_brand": veh_brand.astype(str),
            "region": region,
            "veh_power": veh_power,
            "bonus": bonus,
        }
    )
    return df, {
        "response": "y",
        "offset": "exposure",
        "family": "poisson",
        "terms": {
            "age": {"type": "linear"},
            "veh_brand": {"type": "target_encoding"},
            "region": {"type": "categorical"},
            "veh_power": {"type": "linear"},
        },
    }


def make_gaussian(seed: int = 17) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 20_000
    x1 = rng.standard_normal(n).astype(np.float64)
    x2 = rng.standard_normal(n).astype(np.float64)
    x3 = rng.uniform(0, 10, n).astype(np.float64)
    eta = 2.0 + 0.5 * x1 - 0.3 * x2 + 0.1 * x3**2
    y = eta + rng.normal(0, 0.5, n)
    df = pl.DataFrame(
        {
            "y": y.astype(np.float64),
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )
    return df, {
        "response": "y",
        "family": "gaussian",
        "link": "identity",
        "terms": {
            "x1": {"type": "linear"},
            "x2": {"type": "linear"},
            "x3": {"type": "bs", "df": 4},
        },
    }


def make_binomial(seed: int = 19) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 30_000
    age = rng.uniform(18, 80, n).astype(np.float64)
    income = rng.lognormal(10, 1.0, n).astype(np.float64)
    grade = rng.choice(["A", "B", "C", "D"], size=n, p=[0.2, 0.4, 0.3, 0.1])
    grade_eff = np.where(
        grade == "A", -0.5, np.where(grade == "B", -0.2, np.where(grade == "C", 0.2, 0.5))
    )
    eta = -2.0 + 0.02 * (age - 50) + 0.0001 * (income - 25_000) + grade_eff
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(np.float64)
    df = pl.DataFrame(
        {
            "y": y,
            "age": age,
            "income": income,
            "grade": grade,
        }
    )
    return df, {
        "response": "y",
        "family": "binomial",
        "link": "logit",
        "terms": {
            "age": {"type": "linear"},
            "income": {"type": "linear"},
            "grade": {"type": "categorical"},
        },
    }


def make_gamma(seed: int = 23) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 20_000
    veh_age = rng.uniform(0, 20, n).astype(np.float64)
    region = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    region_eff = np.where(region == "A", 0.0, np.where(region == "B", 0.2, -0.1))
    severity_class = rng.integers(1, 4, n).astype(np.float64)
    eta = 6.0 + 0.05 * veh_age + region_eff + 0.1 * severity_class
    mu = np.exp(eta)
    shape = 2.0
    scale = mu / shape
    y = rng.gamma(shape, scale).astype(np.float64)
    df = pl.DataFrame(
        {
            "y": y,
            "veh_age": veh_age,
            "region": region,
            "severity_class": severity_class,
        }
    )
    return df, {
        "response": "y",
        "family": "gamma",
        "link": "log",
        "terms": {
            "veh_age": {"type": "linear"},
            "region": {"type": "categorical"},
            "severity_class": {"type": "linear"},
        },
    }


def make_edge_tiny(seed: int = 29) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 1_000
    x = rng.standard_normal(n).astype(np.float64)
    eta = 2.0 + 0.7 * x
    y = (eta + rng.normal(0, 0.5, n)).astype(np.float64)
    df = pl.DataFrame({"y": y, "x": x})
    return df, {
        "response": "y",
        "family": "gaussian",
        "link": "identity",
        "terms": {
            "x": {"type": "linear"},
        },
    }


# =============================================================================
# Per-scenario runner
# =============================================================================


def run_scenario(
    label: str,
    df: pl.DataFrame,
    spec: dict,
    unfitted_categorical: list[str] | None = None,
    unfitted_continuous: list[str] | None = None,
    sm_check: bool = False,
) -> CheckCollector:
    c = CheckCollector(label)
    print(f"\n=== {label} ===")
    print(f"  rows={df.shape[0]}, fields={df.shape[1]}, terms={list(spec['terms'].keys())}")
    try:
        result = rs.glm_dict(**spec, data=df).fit()
    except Exception as e:
        c.fail("glm fit", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return c
    print(
        f"  fit: converged={result.converged}, iters={result.iterations}, "
        f"params={len(result.params)}, deviance={result.deviance:.2f}"
    )

    fitted_factors = list(spec["terms"].keys())
    cat_factors = []
    cont_factors = []
    fitted_cat_count = 0
    for name, term in spec["terms"].items():
        ttype = term.get("type")
        if ttype in ("categorical", "target_encoding"):
            cat_factors.append(name)
            fitted_cat_count += 1
        elif ttype in ("linear", "bs", "ns", "ms", "expression"):
            cont_factors.append(name)
        else:
            cont_factors.append(name)
    if unfitted_categorical:
        cat_factors.extend([f for f in unfitted_categorical if f not in cat_factors])
    if unfitted_continuous:
        cont_factors.extend([f for f in unfitted_continuous if f not in cont_factors])

    try:
        diag = result.diagnostics(
            train_data=df,
            categorical_factors=cat_factors,
            continuous_factors=cont_factors,
        )
    except Exception as e:
        c.fail("result.diagnostics", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return c

    # `has_design` controls whether unfitted-factor score tests are expected.
    # When the design matrix isn't stored (lean mode), api.py rebuilds it
    # via `_builder.transform_new_data` (api.py:412-417), so a score test
    # is still produced when a builder is available.
    has_design = (
        result.get_design_matrix() is not None or getattr(result, "_builder", None) is not None
    )
    in_model_set = set(fitted_factors)
    unfitted_set = set((unfitted_categorical or []) + (unfitted_continuous or []))

    # factor_deviance is only populated when at least one categorical factor
    # is fitted (api.py:506: `if compute_deviance_by_level and categorical_factors`).
    # Note: api.py uses the *all* categorical_factors list (fitted + unfitted),
    # so unfitted categorical factors also gate this on. Match that contract.
    expect_factor_dev = len(cat_factors) > 0

    check_top_level_completeness(diag, c, expect_factor_dev=expect_factor_dev)
    check_model_summary(diag, c)
    check_train_diag(diag, c, result=result)
    check_factors(
        diag,
        c,
        expected_in_model=in_model_set,
        expected_unfitted=unfitted_set,
        has_design_matrix=has_design,
    )
    check_lift(diag, c)
    expected_pd = set(cat_factors) | set(cont_factors)
    check_partial_dependence(diag, c, expected_pd)
    check_vif(diag, c)
    check_coefficient_summary(diag, c)
    check_factor_deviance(diag, c)
    check_calibration(diag, c)
    check_json_roundtrip(diag, c)

    # Singular-vs-batch numerical equivalence for compute_factor_significance.
    try:
        check_factor_significance_singular_vs_batch(
            c,
            label,
            result,
            fitted_factors,
        )
    except Exception as e:
        c.fail("factor_significance_singular_vs_batch_setup", f"{type(e).__name__}: {e}")

    # Numerical singular-vs-batch checks: pick the first cat + first cont factor
    # in the data so we have real values to compare on.
    cat_in_data = [f for f in cat_factors if f in df.columns]
    cont_in_data = [f for f in cont_factors if f in df.columns]
    if cat_in_data and cont_in_data:
        family_str = result.family
        # Strip any (...) suffix like 'tweedie(p=1.5)' -> 'tweedie'
        fam_simple = family_str.split("(")[0].strip().lower()
        if "negbinomial" in fam_simple or "negativebinomial" in fam_simple:
            fam_simple = "poisson"  # batch helpers map this internally; safe default
        elif "quasipoisson" in fam_simple:
            fam_simple = "poisson"
        elif "quasibinomial" in fam_simple:
            fam_simple = "binomial"
        try:
            y = df[spec["response"]].to_numpy().astype(np.float64)
            mu = np.asarray(result.fittedvalues, dtype=np.float64)
            exposure = None
            if spec.get("offset") and spec["offset"] in df.columns:
                exposure = df[spec["offset"]].to_numpy().astype(np.float64)
            cat_vals = df[cat_in_data[0]].cast(pl.Utf8).to_numpy()
            cont_vals = df[cont_in_data[0]].to_numpy().astype(np.float64)
            check_singular_vs_batch(c, label, y, mu, exposure, cat_vals, cont_vals, fam_simple)
        except Exception as e:
            c.fail("singular_vs_batch_setup", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    if sm_check and HAS_STATSMODELS:
        try:
            # Lean-mode models store no design matrix; rebuild via the builder.
            X = result.get_design_matrix()
            if X is None and getattr(result, "_builder", None) is not None:
                X = result._builder.transform_new_data(df)
            if X is None:
                c.fail("statsmodels_cross_check", "could not obtain design matrix for cross-check")
            else:
                y = df[spec["response"]].to_numpy().astype(np.float64)
                link_name = result.link
                family_str = result.family.split("(")[0].strip().lower()
                cross_check_statsmodels(family_str, link_name, diag, result, X, y, None, c)
        except Exception as e:
            c.fail("statsmodels_cross_check", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    print(c.report())
    return c


# =============================================================================
# New scenario runners (H, I, J) — added by FIX-K
# =============================================================================


def _expected_top_level_field_names() -> set[str]:
    """Field names that ModelDiagnostics is expected to expose. We don't
    require all to be non-None — just to exist on the dataclass — to confirm
    structural completeness."""
    return {
        "model_summary",
        "train_test",
        "calibration",
        "residual_summary",
        "factors",
        "vif",
        "coefficient_summary",
        "factor_deviance",
        "lift_chart",
        "partial_dependence",
    }


def run_scenario_lean_mode() -> CheckCollector:
    """Scenario H — Lean mode: fit with `store_design_matrix=False` and verify
    that diagnostics still succeed (the api.py fallback rebuilds the design
    matrix via `result._builder.transform_new_data(train_data)`).
    """
    label = "H Lean mode (no design matrix stored)"
    c = CheckCollector(label)
    print(f"\n=== {label} ===")

    rng = np.random.default_rng(101)
    n = 5_000
    age = rng.uniform(20, 70, n).astype(np.float64)
    region = rng.choice(["A", "B", "C"], n)
    exposure = rng.uniform(0.2, 1.0, n).astype(np.float64)
    region_eff = np.where(region == "A", 0.0, np.where(region == "B", 0.15, -0.1))
    eta = -1.5 + 0.01 * (age - 40) + region_eff
    mu = np.exp(eta) * exposure
    y = rng.poisson(mu).astype(np.float64)
    df = pl.DataFrame({"y": y, "exposure": exposure, "age": age, "region": region})
    spec = {
        "response": "y",
        "offset": "exposure",
        "family": "poisson",
        "terms": {
            "age": {"type": "linear"},
            "region": {"type": "categorical"},
        },
    }
    print(f"  rows={df.shape[0]}, store_design_matrix=False")
    try:
        result = rs.glm_dict(**spec, data=df).fit(store_design_matrix=False)
    except Exception as e:
        c.fail("lean.fit", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        print(c.report())
        return c
    print(
        f"  fit: converged={result.converged}, iters={result.iterations}, "
        f"params={len(result.params)}, deviance={result.deviance:.2f}"
    )

    # Confirm the lean-mode preconditions: design matrix is None, builder kept.
    c.check(
        "lean.get_design_matrix() is None",
        result.get_design_matrix() is None,
        "expected lean mode to drop the design matrix",
    )
    c.check(
        "lean._builder available for diagnostic fallback",
        getattr(result, "_builder", None) is not None,
        "lean mode should keep _builder so api.py can rebuild design matrix",
    )

    # Run diagnostics — it must succeed despite the missing X.
    try:
        diag = result.diagnostics(
            train_data=df,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
    except Exception as e:
        c.fail(
            "lean.diagnostics",
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
        print(c.report())
        return c
    c.check("lean.diagnostics returned a value", diag is not None, "diag is None")

    # Top-level structural completeness: factor_deviance is populated (we have
    # a categorical factor), so use the standard checker without overrides.
    check_top_level_completeness(diag, c, expect_factor_dev=True)
    check_model_summary(diag, c)
    check_train_diag(diag, c, result=result)
    # has_design_matrix=True: the api.py fallback at line 412 rebuilds X from
    # the builder, so unfitted-factor score tests are still expected to run.
    check_factors(
        diag,
        c,
        expected_in_model={"age", "region"},
        expected_unfitted=set(),
        has_design_matrix=True,
    )
    check_lift(diag, c)
    check_partial_dependence(diag, c, {"age", "region"})
    check_vif(diag, c)
    check_coefficient_summary(diag, c)
    check_factor_deviance(diag, c)
    check_calibration(diag, c)
    check_json_roundtrip(diag, c)

    # Confirm every expected ModelDiagnostics attribute exists (structural).
    for attr in _expected_top_level_field_names():
        c.check(
            f"lean.diag has attr '{attr}'",
            hasattr(diag, attr),
            f"missing attribute '{attr}' on ModelDiagnostics",
        )

    print(c.report())
    return c


def run_scenario_offset_exposure() -> CheckCollector:
    """Scenario I — Non-trivial offset: exposure ~ Uniform(0.1, 10.0).
    Verifies that A/E ratios are computed in *count* terms (predicted_count =
    rate * exposure), that totals reconcile, and that per-decile exposure_sum
    matches the input exposure sum.
    """
    label = "I Non-trivial exposure offset"
    c = CheckCollector(label)
    print(f"\n=== {label} ===")

    rng = np.random.default_rng(103)
    n = 8_000
    # Wide-dynamic-range exposure makes the count-vs-rate distinction visible.
    exposure = rng.uniform(0.1, 10.0, n).astype(np.float64)
    age = rng.uniform(18, 75, n).astype(np.float64)
    region = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    region_eff = np.where(region == "A", 0.0, np.where(region == "B", 0.2, -0.1))
    eta = -0.7 + 0.012 * (age - 40) + region_eff
    mu = np.exp(eta) * exposure
    y = rng.poisson(mu).astype(np.float64)
    df = pl.DataFrame({"y": y, "exposure": exposure, "age": age, "region": region})
    spec = {
        "response": "y",
        "offset": "exposure",
        "family": "poisson",
        "terms": {
            "age": {"type": "linear"},
            "region": {"type": "categorical"},
        },
    }
    print(
        f"  rows={df.shape[0]}, exposure range=[{exposure.min():.3f}, {exposure.max():.3f}],"
        f" total exposure={exposure.sum():.1f}"
    )
    try:
        result = rs.glm_dict(**spec, data=df).fit()
    except Exception as e:
        c.fail(
            "offset.fit",
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
        print(c.report())
        return c
    print(
        f"  fit: converged={result.converged}, iters={result.iterations},"
        f" deviance={result.deviance:.2f}"
    )

    # Predicted *counts* = rate * exposure → fitted values include exposure.
    fitted = np.asarray(result.fittedvalues, dtype=np.float64)
    sum_y = float(y.sum())
    sum_pred = float(fitted.sum())
    c.check(
        "offset.total_predicted_count ~= total_observed_count",
        math.isclose(sum_pred, sum_y, rel_tol=1e-3, abs_tol=1.0),
        f"sum_y={sum_y:.4f} sum_pred={sum_pred:.4f}",
    )

    try:
        diag = result.diagnostics(
            train_data=df,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
    except Exception as e:
        c.fail(
            "offset.diagnostics",
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
        print(c.report())
        return c

    # train.ae_ratio uses count totals (sum_actual / sum_predicted_count). For
    # a well-fit Poisson it should be ~ 1.0.
    train = diag.train_test.train
    c.check(
        "offset.train.ae_ratio ~ 1.0",
        train.ae_ratio is not None and math.isclose(float(train.ae_ratio), 1.0, abs_tol=0.05),
        f"ae_ratio={train.ae_ratio}",
    )

    # Sum of per-decile exposure should equal the dataset exposure sum.
    decile_exp_sum = sum(d.exposure for d in train.ae_by_decile)
    c.check(
        "offset.sum(decile.exposure) ~= sum(exposure)",
        math.isclose(
            float(decile_exp_sum),
            float(exposure.sum()),
            rel_tol=1e-3,
            abs_tol=1.0,
        ),
        f"sum_decile={decile_exp_sum:.4f} sum_input={exposure.sum():.4f}",
    )

    # Each decile exposure must be > 0 (no rows missing) and finite.
    n_deciles = len(train.ae_by_decile)
    c.check(
        "offset.ae_by_decile has 10 deciles",
        n_deciles == 10,
        f"n={n_deciles}",
    )
    for i, d in enumerate(train.ae_by_decile):
        c.check(
            f"offset.decile[{i}].exposure > 0 and finite",
            _is_finite_float(d.exposure) and d.exposure > 0,
            f"exposure={d.exposure}",
        )

    # `decile.actual` and `decile.predicted` are exposure-weighted *rates*
    # (per unit exposure), so to recover counts we multiply by the decile's
    # exposure and sum across deciles.
    decile_actual_count = sum(d.actual * d.exposure for d in train.ae_by_decile)
    decile_pred_count = sum(d.predicted * d.exposure for d in train.ae_by_decile)
    # Allow slightly larger tolerance because per-decile sums are rounded.
    c.check(
        "offset.sum(decile.actual * exposure) ~= sum(y)",
        math.isclose(
            float(decile_actual_count),
            sum_y,
            rel_tol=5e-3,
            abs_tol=2.0,
        ),
        f"sum_decile_actual_count={decile_actual_count:.4f} sum_y={sum_y:.4f}",
    )
    c.check(
        "offset.sum(decile.predicted * exposure) ~= sum(predicted_count)",
        math.isclose(
            float(decile_pred_count),
            sum_pred,
            rel_tol=5e-3,
            abs_tol=2.0,
        ),
        f"sum_decile_pred_count={decile_pred_count:.4f} sum_pred={sum_pred:.4f}",
    )

    # Standard structural sweep — re-use the existing checkers.
    check_top_level_completeness(diag, c)
    check_model_summary(diag, c)
    check_factors(
        diag,
        c,
        expected_in_model={"age", "region"},
        expected_unfitted=set(),
        has_design_matrix=True,
    )
    check_lift(diag, c)
    check_partial_dependence(diag, c, {"age", "region"})
    check_vif(diag, c)
    check_coefficient_summary(diag, c)
    check_factor_deviance(diag, c)
    check_calibration(diag, c)
    check_json_roundtrip(diag, c)

    print(c.report())
    return c


def run_scenario_unseen_levels() -> CheckCollector:
    """Scenario J — predict() on test data containing an unseen categorical
    level ('D'). Per `interactions.py:_map_to_training_indices` (~line 1620),
    unknown levels map to index 0 (reference category) producing all-zero
    dummy columns. We assert the *current* contract: predict succeeds, returns
    finite values, and downstream diagnostics on this test data still run.
    """
    label = "J Predict on unseen categorical level"
    c = CheckCollector(label)
    print(f"\n=== {label} ===")

    rng = np.random.default_rng(107)
    n_train = 4_000
    age_train = rng.uniform(20, 70, n_train).astype(np.float64)
    region_train = rng.choice(["A", "B", "C"], n_train, p=[0.5, 0.3, 0.2])
    exposure_train = rng.uniform(0.2, 1.0, n_train).astype(np.float64)
    region_eff = np.where(
        region_train == "A",
        0.0,
        np.where(region_train == "B", 0.2, -0.1),
    )
    eta = -1.0 + 0.01 * (age_train - 40) + region_eff
    mu = np.exp(eta) * exposure_train
    y_train = rng.poisson(mu).astype(np.float64)
    train = pl.DataFrame(
        {
            "y": y_train,
            "exposure": exposure_train,
            "age": age_train,
            "region": region_train,
        }
    )
    spec = {
        "response": "y",
        "offset": "exposure",
        "family": "poisson",
        "terms": {
            "age": {"type": "linear"},
            "region": {"type": "categorical"},
        },
    }
    print(f"  train rows={train.shape[0]}, train levels=A/B/C")
    try:
        result = rs.glm_dict(**spec, data=train).fit()
    except Exception as e:
        c.fail(
            "unseen.fit",
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
        print(c.report())
        return c
    print(f"  fit: converged={result.converged}, deviance={result.deviance:.2f}")

    # Build test data containing an unseen level 'D'.
    n_test = 800
    age_test = rng.uniform(20, 70, n_test).astype(np.float64)
    region_test = rng.choice(["A", "B", "C", "D"], n_test, p=[0.3, 0.3, 0.2, 0.2])
    exposure_test = rng.uniform(0.2, 1.0, n_test).astype(np.float64)
    y_test = rng.poisson(np.exp(-0.5) * exposure_test).astype(np.float64)
    test = pl.DataFrame(
        {
            "y": y_test,
            "exposure": exposure_test,
            "age": age_test,
            "region": region_test,
        }
    )
    n_unseen = int((np.asarray(region_test) == "D").sum())
    c.check(
        "unseen.test contains at least one 'D' row",
        n_unseen > 0,
        f"n_unseen_D={n_unseen}",
    )
    print(f"  test rows={test.shape[0]}, unseen 'D' rows={n_unseen}")

    # The contract (per interactions.py:_map_to_training_indices, default=0):
    # predict() succeeds and silently maps unseen levels to the reference.
    predict_raised: BaseException | None = None
    preds = None
    try:
        preds = result.predict(test)
    except Exception as e:
        predict_raised = e

    c.check(
        "unseen.predict() succeeds (current contract: silent fallback)",
        predict_raised is None,
        f"raised {type(predict_raised).__name__}: {predict_raised}",
    )

    if predict_raised is None:
        preds_arr = np.asarray(preds, dtype=np.float64)
        c.check(
            "unseen.predict shape == n_test",
            preds_arr.shape == (n_test,),
            f"shape={preds_arr.shape}",
        )
        c.check(
            "unseen.predict all finite",
            bool(np.all(np.isfinite(preds_arr))),
            f"non-finite count={int(np.sum(~np.isfinite(preds_arr)))}",
        )
        c.check(
            "unseen.predict all > 0 (Poisson with log link)",
            bool(np.all(preds_arr > 0)),
            f"non-positive count={int(np.sum(preds_arr <= 0))}",
        )

        # Confirm unseen-level rows are predicted *as if* they were the
        # reference level 'A' (all dummy columns 0). We check by comparing
        # predictions on D-rows against the same rows with region overwritten
        # to 'A' — they should be identical.
        test_as_A = test.with_columns(
            pl.when(pl.col("region") == "D")
            .then(pl.lit("A"))
            .otherwise(pl.col("region"))
            .alias("region")
        )
        try:
            preds_as_A = np.asarray(result.predict(test_as_A), dtype=np.float64)
            mask_D = np.asarray(region_test) == "D"
            max_diff = (
                float(np.max(np.abs(preds_arr[mask_D] - preds_as_A[mask_D])))
                if mask_D.any()
                else 0.0
            )
            c.check(
                "unseen.D rows predicted as reference (A)",
                max_diff < 1e-9,
                f"max diff between predict(D) and predict(A) = {max_diff:.6e}",
            )
        except Exception as e:
            c.fail(
                "unseen.compare_to_reference",
                f"{type(e).__name__}: {e}",
            )

        # Diagnostics on the *original* train_data should still work even
        # though predict() was successfully called on test data containing
        # unseen levels — the model's internal state hasn't been mutated.
        try:
            diag_train = result.diagnostics(
                train_data=train,
                categorical_factors=["region"],
                continuous_factors=["age"],
            )
            c.check(
                "unseen.diagnostics(train) still works after predict(test)",
                diag_train is not None,
                "diag is None",
            )
            check_top_level_completeness(diag_train, c)
            check_model_summary(diag_train, c)
            check_factors(
                diag_train,
                c,
                expected_in_model={"age", "region"},
                expected_unfitted=set(),
                has_design_matrix=True,
            )
            check_lift(diag_train, c)
            check_calibration(diag_train, c)
            check_json_roundtrip(diag_train, c)
        except Exception as e:
            c.fail(
                "unseen.diagnostics_on_train",
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

        # Also call diagnostics with the unseen-level test data passed via
        # `test_data=` (the holdout slot). This is the supported path for
        # evaluating a model on differently-sized data — it triggers
        # transform_new_data() on the test frame and predicts fresh, rather
        # than reusing the model's fittedvalues. We assert it completes
        # without raising. If diagnostics(train_data=test_data,...) where
        # test_data has a different row count panics in the underlying
        # residual computation (mismatch between stored fittedvalues and
        # supplied train_data), that's a production bug to report — not
        # something this harness should mask.
        diag_test_raised: BaseException | None = None
        try:
            diag_test = result.diagnostics(
                train_data=train,
                categorical_factors=["region"],
                continuous_factors=["age"],
                test_data=test,
            )
        except Exception as e:
            diag_test_raised = e
        c.check(
            "unseen.diagnostics(test_data=...) handles unseen levels",
            diag_test_raised is None,
            f"raised {type(diag_test_raised).__name__}: {diag_test_raised}",
        )
        if diag_test_raised is None:
            c.check(
                "unseen.diagnostics test_data populated train_test.test",
                diag_test.train_test is not None and diag_test.train_test.test is not None,
                "train_test.test is None",
            )
    else:
        # Alternative contract path: clear ValidationError. We don't expect
        # this today, but if the codebase changes to raise, the harness
        # requires the error type to be ValidationError (per the spec).
        from rustystats.exceptions import ValidationError

        c.check(
            "unseen.predict raises ValidationError (alternative contract)",
            isinstance(predict_raised, ValidationError),
            f"raised {type(predict_raised).__name__}",
        )

    print(c.report())
    return c


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    print("=" * 70)
    print("RustyStats diagnostics correctness verification harness")
    print("=" * 70)
    print(f"statsmodels available: {HAS_STATSMODELS}")

    collectors: list[CheckCollector] = []

    # Scenario A: small Poisson with offset, mix of linear/cat/spline
    df, spec = make_poisson_small()
    collectors.append(
        run_scenario(
            "A small Poisson + offset",
            df,
            spec,
            unfitted_categorical=[],  # region already in model
            unfitted_continuous=["extra_cont"],
            sm_check=False,  # spline + offset complicates direct sm comparison
        )
    )

    # Scenario B: Poisson + target encoding
    df, spec = make_poisson_with_te()
    collectors.append(
        run_scenario(
            "B Poisson + TE high-card",
            df,
            spec,
            unfitted_continuous=["bonus"],
            sm_check=False,  # TE has no statsmodels equivalent
        )
    )

    # Scenario C: Gaussian (good cross-check candidate)
    df, spec = make_gaussian()
    collectors.append(
        run_scenario(
            "C Gaussian + spline",
            df,
            spec,
            sm_check=False,  # spline blocks direct sm comparison
        )
    )

    # Scenario D: Binomial (good cross-check candidate, no spline)
    df, spec = make_binomial()
    collectors.append(
        run_scenario(
            "D Binomial logit",
            df,
            spec,
            sm_check=True,
        )
    )

    # Scenario E: Gamma severity-style
    df, spec = make_gamma()
    collectors.append(
        run_scenario(
            "E Gamma severity",
            df,
            spec,
            sm_check=True,
        )
    )

    # Scenario F: tiny dataset, single feature
    df, spec = make_edge_tiny()
    collectors.append(
        run_scenario(
            "F Edge tiny 1-feature",
            df,
            spec,
            sm_check=True,
        )
    )

    # Model serialization round-trip on Scenario A's setup
    print("\n=== Model serialization round-trip ===")
    df, spec = make_poisson_small()
    cs = CheckCollector("G serialize+predict")
    try:
        result = rs.glm_dict(**spec, data=df).fit()
        check_model_serialization(result, df, cs)
    except Exception as e:
        cs.fail("serialize_setup", f"{type(e).__name__}: {e}")
    print(cs.report())
    collectors.append(cs)

    # Scenario H: lean mode (no design matrix stored). The `compute_diagnostics`
    # path must rebuild the design matrix from the train_data via
    # `result._builder.transform_new_data(...)` (see api.py around line 412).
    collectors.append(run_scenario_lean_mode())

    # Scenario I: Poisson with a non-trivial exposure offset (uniform [0.1, 10]).
    # Verifies A/E ratios use exposure (not just count), the predicted-count
    # totals reconcile, and per-decile exposure_sum totals match the input.
    collectors.append(run_scenario_offset_exposure())

    # Scenario J: predict on a test frame containing an unseen categorical level.
    # The codebase silently maps unseen levels to index 0 (reference) — see
    # `interactions.py:1620-1638` `_map_to_training_indices` (default=0). This
    # scenario asserts that contract and that downstream diagnostics still run.
    collectors.append(run_scenario_unseen_levels())

    # Aggregate
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    total = sum(c.n_total for c in collectors)
    failed = sum(c.n_failed for c in collectors)
    passed = sum(c.n_passed for c in collectors)
    print(f"Scenarios run:     {len(collectors)}")
    print(f"Total invariants:  {total}")
    print(f"Passed:            {passed}")
    print(f"Failed:            {failed}")
    print()
    if failed == 0:
        print("OVERALL: PASS - no diagnostic regressions detected.")
        return 0
    else:
        print("OVERALL: FAIL - regressions detected. See details above.")
        # Print just the failing checks for easy scanning
        print("\nFailing checks:")
        for c in collectors:
            for r in c.results:
                if not r.passed:
                    print(f"  [{c.scenario}] {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
