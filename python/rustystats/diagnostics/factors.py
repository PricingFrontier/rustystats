"""
Factor-level diagnostic computations.

_FactorDiagnosticsComputer handles per-factor analysis including:
- Actual vs Expected (A/E) by level/bin
- Residual patterns
- Factor significance (Wald chi-square)
- Score tests for unfitted factors
- Coefficient extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    chi2_cdf_py as _chi2_cdf,
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
    compute_factor_significance_batch_py as _rust_factor_significance_batch,
)
from rustystats._rustystats import (
    compute_residual_pattern_batch_py as _rust_residual_pattern_batch,
)
from rustystats._rustystats import (
    compute_residual_pattern_py as _rust_residual_pattern,
)
from rustystats._rustystats import (
    score_test_continuous_batch_py as _rust_score_test_continuous_batch,
)
from rustystats._rustystats import (
    score_test_continuous_py as _rust_score_test_continuous,
)
from rustystats._rustystats import (
    t_cdf_py as _t_cdf,
)
from rustystats.constants import (
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    EPSILON,
)
from rustystats.diagnostics.types import (
    ActualExpectedBin,
    FactorCoefficient,
    FactorDiagnostics,
    FactorSignificance,
    ResidualPattern,
    ScoreTestResult,
)
from rustystats.diagnostics.utils import validate_factor_in_data
from rustystats.exceptions import FittingError

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Factor → feature mapping (single source of truth)
#
# Five sites in this module previously used `if name in fn` substring matching
# to find features for a variable. That produces false positives when one
# variable name is a substring of another (e.g. factor `Age` matches
# `bs(VehAge, 1/4)`). The pattern below uses STRICT delimiter-based matching
# and is shared via `_FactorFeatureIndex` so the bug class cannot reappear.
# =============================================================================


# Spline-style transform prefixes recognized everywhere in this module.
_SPLINE_PREFIXES: tuple[str, ...] = ("bs(", "ns(", "ms(", "s(")


@dataclass
class _FactorFeature:
    """Per-factor view: which feature_names belong to this factor.

    Attributes
    ----------
    indices
        Positions in ``feature_names`` (the list passed to the index) where
        this variable appears.
    feature_names
        The feature_names themselves, in the same order as ``indices``.
    term_type
        Coarse classification of the term: ``"linear"``, ``"spline"``,
        ``"expression"``, ``"te"``, ``"categorical"``, ``"interaction"``,
        ``"other"``, or ``"unknown"`` if the variable is not in the model.
        When a variable contributes to multiple kinds (e.g. a linear term
        plus a spline) the kind of the first matched feature is recorded.
    transformation
        A representative feature name for display (the first matched feature),
        or ``None`` if the variable is not in the model.
    """

    indices: list[int] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    term_type: str = "unknown"
    transformation: str | None = None


def _match_factor(feature_name: str, var: str) -> tuple[bool, str]:
    """Return (matched, term_kind) using STRICT delimiter-based checks.

    Avoids the substring-matching bug where ``Age`` matches ``VehAge`` by
    requiring the variable name to be bounded by an appropriate delimiter
    (parenthesis, comma, colon, or end-of-string) for transform-style features
    and by a regex word boundary for raw-expression features.

    The recognized term kinds are:

    - ``"linear"``       — exact match (e.g. ``var``).
    - ``"spline"``       — ``bs(var,...)``, ``ns(var,...)``, ``s(var,...)``,
      ``ms(var,...)`` where ``var`` is followed by ``,`` or ``)``.
    - ``"te"``           — ``TE(var)`` or ``TE(...:var:...)`` (target encoded).
    - ``"categorical"``  — ``C(var)[...]`` (treatment encoding).
    - ``"expression"``   — ``I(...)`` containing ``var`` as a whole word.
    - ``"interaction"``  — feature contains ``:`` and at least one part
      matches ``var`` recursively.
    - ``"other"``        — fallback word-boundary match for transforms not
      explicitly named above (e.g. ``log(var)``), excluding interactions.
    - ``"unknown"``      — no match.
    """
    if feature_name == "Intercept":
        return False, "unknown"

    # Linear (exact match).
    if feature_name == var:
        return True, "linear"

    # Spline-style transforms: bs(var,...), ns(var,...), s(var,...), ms(var,...).
    for prefix in _SPLINE_PREFIXES:
        if feature_name.startswith(prefix):
            inner = feature_name[len(prefix) :]
            if inner.startswith(var):
                next_char_idx = len(var)
                if next_char_idx < len(inner) and inner[next_char_idx] in (",", ")"):
                    return True, "spline"
            # Found a spline prefix but the leading argument did not match
            # `var` — short-circuit. (`var` could still appear later via the
            # interaction branch below.)
            break

    # Target encoding: TE(var) exact-or-wrapped, or TE(part:part...) where any
    # part equals `var`. The closing ")" (if present) is stripped before split.
    if feature_name.startswith("TE("):
        # Drop the leading "TE(" and an optional trailing ")".
        inner = feature_name[3:]
        if inner.endswith(")"):
            inner = inner[:-1]
        for part in inner.split(":"):
            if part == var:
                return True, "te"

    # Categorical treatment encoding: C(var)[...].
    if feature_name.startswith("C("):
        inner = feature_name[2:]
        if inner.startswith(var):
            next_char_idx = len(var)
            if next_char_idx < len(inner) and inner[next_char_idx] == ")":
                return True, "categorical"

    # Identity expression: I(<expr>) where var appears as a whole word.
    if feature_name.startswith("I(") and feature_name.endswith(")"):
        expr = feature_name[2:-1]
        pattern = re.compile(rf"\b{re.escape(var)}\b")
        if pattern.search(expr):
            return True, "expression"

    # Interaction features: contain ":" and at least one part matches `var`.
    # Each part may itself be a spline / TE / linear feature, so recurse on it.
    if ":" in feature_name:
        for part in feature_name.split(":"):
            matched, _kind = _match_factor(part, var)
            if matched:
                return True, "interaction"

    # Fallback: word-boundary match for other transforms (log(var), sqrt(var),
    # etc.) that aren't explicitly recognized above. Excludes interactions
    # (already handled) and exact equality (already handled).
    if ":" not in feature_name:
        pattern = re.compile(rf"\b{re.escape(var)}\b")
        if pattern.search(feature_name):
            return True, "other"

    return False, "unknown"


class _FactorFeatureIndex:
    """Pre-computed mapping from factor names to their feature-name positions.

    Built once with the full set of variable names + the model's feature_names
    list, then queried via :py:meth:`features_for` and :py:meth:`is_in_model`.
    Each lookup is O(1) — there is no per-call substring scan.
    """

    def __init__(self, all_factor_names: list[str], feature_names: list[str]):
        self._index: dict[str, _FactorFeature] = {}
        for var in all_factor_names:
            self._index[var] = self._build_for_var(var, feature_names)

    @staticmethod
    def _build_for_var(var: str, feature_names: list[str]) -> _FactorFeature:
        indices: list[int] = []
        names: list[str] = []
        term_type: str | None = None

        for i, fn in enumerate(feature_names):
            if fn == "Intercept":
                continue
            matched, kind = _match_factor(fn, var)
            if matched:
                indices.append(i)
                names.append(fn)
                if term_type is None:
                    term_type = kind
                # If the variable appears in multiple kinds (e.g. a linear
                # term AND a spline) we keep the kind of the first match.

        return _FactorFeature(
            indices=indices,
            feature_names=names,
            term_type=term_type or "unknown",
            transformation=names[0] if names else None,
        )

    def features_for(self, var: str) -> _FactorFeature:
        """Return the (cached) ``_FactorFeature`` for ``var``.

        Returns an empty ``_FactorFeature`` for variables not registered at
        construction time.
        """
        cached = self._index.get(var)
        if cached is not None:
            return cached
        return _FactorFeature()

    def is_in_model(self, var: str) -> bool:
        """True iff at least one feature_name was matched for ``var``."""
        return bool(self.features_for(var).indices)


class _FactorDiagnosticsComputer:
    """Computes per-factor diagnostics for fitted GLM models.

    Requires arrays from the parent DiagnosticsComputer: y, mu, exposure,
    pearson_residuals, feature_names, and family string.
    """

    def __init__(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        pearson_residuals: np.ndarray,
        feature_names: list[str],
        family: str,
        weights: np.ndarray | None = None,
        base_mu: np.ndarray | None = None,
    ):
        self.y = y
        self.mu = mu
        self.exposure = exposure
        self._has_weights = weights is not None
        self.weights = (
            np.asarray(weights, dtype=np.float64) if weights is not None else np.ones_like(y)
        )
        self.base_mu = None if base_mu is None else np.asarray(base_mu, dtype=np.float64)
        self.pearson_residuals = pearson_residuals
        self.feature_names = feature_names
        self.family = family
        # Cache the feature_names as a list and a set. Set is used for O(1)
        # exact-match checks; list is iterated for substring/prefix checks.
        self._feature_names_list = list(feature_names)
        self._feature_set = set(feature_names)
        # Per-variable feature mapping cache. Built lazily on first lookup per
        # variable; survives across all callers (the orchestrator's initial
        # pass through `compute_factor_diagnostics` plus the singular methods
        # like `_is_in_model`, `_get_factor_coefficients`, and
        # `_compute_eta_contribution`).
        #
        # Replaces an earlier `frozenset(all_factor_names)`-keyed cache that
        # thrashed when the orchestrator populated it with the full factor
        # set and the singular methods then asked for just `[var]`: the
        # cache key mismatched, the entire index was rebuilt for one
        # variable, and the cache was overwritten — repeating 30+ times.
        self._var_feature_cache: dict[str, _FactorFeature] = {}

    @property
    def _prior_weights(self) -> np.ndarray | None:
        """Prior-weight array to forward to the Rust A/E kernels, or ``None``
        when the fit was unweighted (so the kernel uses unit weights)."""
        return self.weights if self._has_weights else None

    def _get_feature_for(self, var: str) -> _FactorFeature:
        """Return the ``_FactorFeature`` for ``var``, building on first request.

        The result is memoized in ``self._var_feature_cache``, so subsequent
        calls for the same variable are O(1). Each variable's mapping is
        constructed independently the first time it is asked for; there is
        no eager bulk build.

        Defensive against partially-initialized instances (the verify harness
        constructs a computer via ``__new__`` and only attaches
        ``feature_names``): the cache and ``_feature_names_list`` are
        synthesized on demand when missing.
        """
        cache = getattr(self, "_var_feature_cache", None)
        if cache is None:
            cache = {}
            self._var_feature_cache = cache

        cached = cache.get(var)
        if cached is not None:
            return cached

        feature_names_list = getattr(self, "_feature_names_list", None)
        if feature_names_list is None:
            feature_names_list = list(getattr(self, "feature_names", []))
            self._feature_names_list = feature_names_list

        feature = _FactorFeatureIndex._build_for_var(var, feature_names_list)
        cache[var] = feature
        return feature

    def compute_factor_diagnostics(
        self,
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
        result=None,
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        design_matrix: np.ndarray | None = None,
        bread_matrix: np.ndarray | None = None,
        irls_weights: np.ndarray | None = None,
        cat_column_cache: dict | None = None,
        cont_column_cache: dict | None = None,
        cat_unique_cache: dict | None = None,
        compute_score_tests: bool = True,
    ) -> list[FactorDiagnostics]:
        """Compute diagnostics for each specified factor.

        For unfitted factors, computes Rao's score test if design_matrix,
        bread_matrix, and irls_weights are provided.
        """
        factors = []

        can_compute_score_test = (
            compute_score_tests
            and design_matrix is not None
            and bread_matrix is not None
            and irls_weights is not None
        )

        # All downstream lookups go through `self._get_feature_for(var)`,
        # which lazily builds and caches per-variable mappings. Each variable
        # is processed once on first request; subsequent calls (significance,
        # coefficients, transformation, `_compute_eta_contribution`) reuse
        # the cached entry. This avoids both eager bulk construction and the
        # earlier `frozenset` cache-thrashing pattern.

        # Pre-compute factor significance for ALL fitted factors in a single
        # batched Rust call (parallelized over factors via rayon). This replaces
        # the per-factor np.linalg.inv loop that previously dominated the
        # significance phase of compute_factor_diagnostics. The downstream
        # per-factor passes look up `significance_lookup[name]` instead of
        # invoking `compute_factor_significance` per factor; the singular
        # method is preserved for direct external callers.
        significance_lookup: dict[str, FactorSignificance | None] = {}
        if (
            bread_matrix is not None
            and result is not None
            and hasattr(result, "params")
            and hasattr(result, "bse")
        ):
            # Gather one entry per factor (categorical first, continuous second)
            # in a stable order matching the per-factor loops below. Only
            # factors that are `in_model` need a significance entry; for the
            # rest we leave `significance_lookup[name]` unset so the per-factor
            # loops fall back to None as before.
            all_factor_names: list[str] = []
            param_indices_per_factor: list[list[int]] = []
            for name in list(categorical_factors) + list(continuous_factors):
                # Use the strict-matching per-variable cache (avoids substring
                # false positives such as `Age` matching `bs(VehAge, 1/4)`).
                feat = self._get_feature_for(name)
                if not feat.indices:
                    continue
                all_factor_names.append(name)
                param_indices_per_factor.append(list(feat.indices))

            if param_indices_per_factor:
                try:
                    params_arr = np.asarray(result.params, dtype=np.float64)
                    bse_attr = result.bse
                    bse_arr = np.asarray(
                        bse_attr() if callable(bse_attr) else bse_attr,
                        dtype=np.float64,
                    )
                    bread_arr = np.ascontiguousarray(bread_matrix, dtype=np.float64)
                    raw_results = _rust_factor_significance_batch(
                        param_indices_per_factor, params_arr, bse_arr, bread_arr
                    )
                    for fname, raw in zip(all_factor_names, raw_results):
                        if raw is None:
                            significance_lookup[fname] = None
                        else:
                            chi2 = float(raw["chi2"])
                            pvalue = float(raw["pvalue"])
                            # Mirror the singular path's rounding contract:
                            # chi2 -> 2dp, p -> 4dp, dev_contrib == chi2 -> 2dp.
                            significance_lookup[fname] = FactorSignificance(
                                chi2=round(chi2, 2),
                                p=round(pvalue, 4),
                                dev_contrib=round(chi2, 2),
                            )
                except AttributeError:
                    significance_lookup = {fname: None for fname in all_factor_names}
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    # Optional optimization: the batched Rust call is a perf
                    # path that is mathematically identical to invoking the
                    # singular `compute_factor_significance` once per factor.
                    # If the batch call fails for a numerical reason (e.g. a
                    # singular sub-matrix), we fall back to the per-factor
                    # singular path below — it will surface the same failure
                    # via its own (narrow) raise if the underlying problem is
                    # genuine. Resetting the lookup to {} causes the downstream
                    # `name in significance_lookup` check to miss and dispatch
                    # to `self.compute_factor_significance(...)` per factor.
                    import logging

                    logging.getLogger(__name__).warning(
                        "Batched factor significance failed (%s); falling back "
                        "to per-factor computation which produces identical results.",
                        e,
                    )
                    significance_lookup = {}

        # Process categorical factors.
        #
        # Two-pass structure analogous to the continuous block below: in PASS 1
        # we collect per-factor values, the cached (unique, inverse) pair, and
        # any u32 codes available from cat_unique_cache; in PASS 2 we assemble
        # FactorDiagnostics entries. Between the passes we issue a single
        # batched A/E Rust call for all factors that have a code entry in the
        # cache, replacing the k sequential 1M-element string tolist()
        # materializations that previously dominated wall-clock.
        cat_entries: list[dict] = []
        for name in categorical_factors:
            validate_factor_in_data(name, data, "Categorical factor")

            values = (
                cat_column_cache[name]
                if cat_column_cache and name in cat_column_cache
                else data[name].to_numpy().astype(str)
            )
            in_model = bool(self._get_feature_for(name).indices)

            # Use pre-computed unique/inverse from cache or compute. When the
            # api.py cache is populated it ships both `sorted_levels` and the
            # matching u32 `codes` — reuse them for the batched A/E call.
            codes = None
            if cat_unique_cache and name in cat_unique_cache:
                unique, inverse = cat_unique_cache[name]
                # The cache stores (sorted_levels: np.ndarray[str],
                # codes: np.ndarray[uint32]). The A/E path wants u32 codes;
                # the residual-pattern path wants (unique, inverse). These
                # happen to be the same shape because `inverse` in the cache
                # IS the uint32 code array. Use it directly.
                codes = np.ascontiguousarray(inverse, dtype=np.uint32)
            else:
                unique, inverse = np.unique(values, return_inverse=True)

            cat_entries.append(
                {
                    "name": name,
                    "values": values,
                    "in_model": in_model,
                    "unique": unique,
                    "inverse": inverse,
                    "codes": codes,
                }
            )

        # BATCH CALL: collect per-factor code arrays and dispatch a single
        # Rust call that runs the k per-factor A/E computations in parallel
        # via rayon. Factors without a cached codes array fall back to the
        # singular string-based path below.
        #
        # Memory: pass a Python list of contiguous uint32 arrays (one per
        # factor) instead of stacking them into a single (n, k) matrix. The
        # Rust binding zero-copies each into a `&[u32]` slice. Skipping the
        # matrix allocation saves ~24 MB transient peak at 1M rows × 6 cat
        # factors (and the same again on the Rust side that previously did
        # `column(j).to_vec()`).
        ae_bins_per_factor: list[list[ActualExpectedBin] | None] = [None for _ in cat_entries]
        batchable_indices = [i for i, e in enumerate(cat_entries) if e["codes"] is not None]
        if batchable_indices:
            codes_list: list[np.ndarray] = []
            levels_list: list[list[str]] = []
            for src_idx in batchable_indices:
                entry = cat_entries[src_idx]
                codes_list.append(entry["codes"])
                # `unique` is an ndarray of strings; the Rust binding needs a
                # Python list[str]. tolist() on a k-element array is cheap.
                levels_list.append([str(v) for v in entry["unique"]])

            batch_result = _rust_ae_categorical_batch(
                codes_list,
                levels_list,
                self.y,
                self.mu,
                self.exposure,
                rare_threshold_pct,
                max_categorical_levels,
                self.family,
                prior_weights=self._prior_weights,
                base=self.base_mu,
            )
            for out_col, src_idx in enumerate(batchable_indices):
                ae_bins_per_factor[src_idx] = self._format_ae_bins(batch_result[out_col])

        # PASS 2: assemble each factor's FactorDiagnostics. Use the batched A/E
        # result where available; otherwise fall back to the singular path.
        for idx, entry in enumerate(cat_entries):
            name = entry["name"]
            values = entry["values"]
            in_model = entry["in_model"]
            unique = entry["unique"]
            inverse = entry["inverse"]

            if ae_bins_per_factor[idx] is not None:
                ae_bins = ae_bins_per_factor[idx]
            else:
                ae_bins = self._compute_ae_categorical(
                    values, rare_threshold_pct, max_categorical_levels
                )
            resid_pattern = self._compute_residual_pattern_categorical(
                values, precomputed_unique_inverse=(unique, inverse)
            )
            # Look up the precomputed batched significance result; fall back to
            # the singular path only when the batch call was skipped (e.g. no
            # bread matrix or fitted result, or the batch call raised).
            if in_model and result:
                if name in significance_lookup:
                    significance = significance_lookup[name]
                else:
                    significance = self.compute_factor_significance(name, result, bread_matrix)
            else:
                significance = None
            coefficients = (
                self._get_factor_coefficients(name, result) if in_model and result else None
            )

            score_test = None
            if not in_model and can_compute_score_test:
                score_test = self._compute_score_test_categorical(
                    values, design_matrix, bread_matrix, irls_weights
                )

            factors.append(
                FactorDiagnostics(
                    name=name,
                    factor_type="categorical",
                    in_model=in_model,
                    transform=self._get_feature_for(name).transformation,
                    coefficients=coefficients,
                    actual_vs_expected=ae_bins,
                    residual_pattern=resid_pattern,
                    significance=significance,
                    score_test=score_test,
                )
            )

        # Process continuous factors.
        #
        # First pass: precompute per-factor `values` and `in_model`, and stage
        # the batched score-test inputs so we only invoke the Rust solver once
        # for the entire group of unfitted continuous factors.
        cont_entries: list[dict] = []
        score_test_zs: list[np.ndarray] = []
        score_test_indices: list[int] = []
        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")

            values = (
                cont_column_cache[name]
                if cont_column_cache and name in cont_column_cache
                else data[name].to_numpy().astype(np.float64)
            )
            in_model = bool(self._get_feature_for(name).indices)

            # Compute the finite mask once per factor and reuse it both for
            # score-test NaN/Inf scrubbing (when applicable) and for the
            # PASS 2 residual-pattern fallback. is_valid_any caches whether
            # any entry is finite so PASS 2 doesn't have to scan again.
            finite_mask = np.isfinite(values)
            is_valid_any = bool(finite_mask.any())

            entry = {
                "name": name,
                "values": values,
                "in_model": in_model,
                "is_valid_any": is_valid_any,
            }
            cont_entries.append(entry)

            if not in_model and can_compute_score_test:
                # Apply the same NaN/Inf scrubbing that the singular path does
                # (replace invalid entries with the mean of the valid ones).
                if finite_mask.all():
                    z = values
                else:
                    z = values.copy()
                    mean_val = float(np.mean(values[finite_mask])) if is_valid_any else 0.0
                    z[~finite_mask] = mean_val
                score_test_zs.append(np.asarray(z, dtype=np.float64))
                score_test_indices.append(len(cont_entries) - 1)

        # Run the batched score test if any unfitted continuous factors were
        # collected, then attach the per-column result back to the matching
        # entry in `cont_entries`.
        score_test_results: dict[int, ScoreTestResult] = {}
        if score_test_zs:
            try:
                # Preallocate (n, k) directly instead of using np.column_stack,
                # which would both construct a stacked array AND then re-copy
                # via astype. One allocation + one copy-per-column is cheaper.
                n = score_test_zs[0].shape[0]
                zs = np.empty((n, len(score_test_zs)), dtype=np.float64)
                for j, z in enumerate(score_test_zs):
                    zs[:, j] = z
                batch_results = _rust_score_test_continuous_batch(
                    zs,
                    design_matrix,
                    self.y,
                    self.mu,
                    irls_weights,
                    bread_matrix,
                    self.family,
                )
                for idx, raw in zip(score_test_indices, batch_results):
                    score_test_results[idx] = ScoreTestResult(
                        statistic=round(raw["statistic"], 2),
                        df=raw["df"],
                        pvalue=round(raw["pvalue"], 4),
                        significant=raw["significant"],
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # Optional optimization: the batched Rust call shares precomputed
                # quantities across the k columns of `zs`. If it fails for a
                # numerical reason, fall back to invoking the singular score
                # test per column — same math, same Rust kernel, just slower.
                # Resetting `score_test_results` to {} would silently mask the
                # failure (downstream `score_test_results.get(idx)` returns
                # None), so we re-run via the singular path which will raise
                # its own (narrow) exception if the underlying problem is real.
                import logging

                logging.getLogger(__name__).warning(
                    "Batched score test failed (%s); falling back to per-factor "
                    "calls which produce identical results.",
                    e,
                )
                score_test_results = {}
                for idx, z in zip(score_test_indices, score_test_zs):
                    singular_result = self._compute_score_test_continuous(
                        z, design_matrix, bread_matrix, irls_weights
                    )
                    if singular_result is not None:
                        score_test_results[idx] = singular_result

        # Batch the continuous A/E and residual-pattern Rust calls across all
        # collected factors. Each of these is embarrassingly parallel over k
        # factors; a single Rust call with rayon-parallel internals is much
        # faster than k sequential single-threaded calls, especially for large
        # n and moderate k.
        #
        # Memory: pass a Python list of the per-factor `values` arrays directly
        # instead of stacking them into an (n, k) `values_matrix`. Each entry
        # is already a contiguous numpy array; the Rust binding zero-copies it
        # into a `&[f64]` slice. Skipping the matrix allocation saves a 192 MB
        # transient peak at 1M rows × 24 continuous factors (and the same
        # again on the Rust side, which previously did `column(j).to_vec()`).
        ae_bins_per_factor: list[list[ActualExpectedBin]] = [[] for _ in cont_entries]
        resid_patterns_per_factor: list[ResidualPattern] = [
            ResidualPattern(resid_corr=0.0, var_explained=0.0) for _ in cont_entries
        ]
        if cont_entries:
            values_list = [
                np.ascontiguousarray(entry["values"], dtype=np.float64) for entry in cont_entries
            ]

            ae_batch = _rust_ae_continuous_batch(
                values_list,
                self.y,
                self.mu,
                self.exposure,
                n_bins,
                self.family,
                prior_weights=self._prior_weights,
                base=self.base_mu,
            )
            for i, rust_bins in enumerate(ae_batch):
                ae_bins_per_factor[i] = self._format_ae_bins(rust_bins)

            # Per-factor valid-mask short-circuit (preserves singular-path
            # semantics: factors whose values are entirely NaN/Inf get a
            # zero-filled pattern rather than whatever Rust would produce).
            # `is_valid_any` was cached in PASS 1 — no need to scan again.
            rp_batch = _rust_residual_pattern_batch(values_list, self.pearson_residuals, n_bins)
            for i, raw in enumerate(rp_batch):
                if cont_entries[i]["is_valid_any"]:
                    resid_patterns_per_factor[i] = self._format_residual_pattern_continuous(raw)
            # PASS 2 doesn't read `values_list`, so free it early — the
            # entry["values"] references still keep the arrays alive.
            del values_list

        # Second pass: assemble FactorDiagnostics now that score tests are known.
        for idx, entry in enumerate(cont_entries):
            name = entry["name"]
            in_model = entry["in_model"]

            ae_bins = ae_bins_per_factor[idx]
            resid_pattern = resid_patterns_per_factor[idx]
            # Look up the precomputed batched significance result; fall back to
            # the singular path only when the batch call was skipped (e.g. no
            # bread matrix or fitted result, or the batch call raised).
            if in_model and result:
                if name in significance_lookup:
                    significance = significance_lookup[name]
                else:
                    significance = self.compute_factor_significance(name, result, bread_matrix)
            else:
                significance = None
            coefficients = (
                self._get_factor_coefficients(name, result) if in_model and result else None
            )

            score_test = score_test_results.get(idx)

            factors.append(
                FactorDiagnostics(
                    name=name,
                    factor_type="continuous",
                    in_model=in_model,
                    transform=self._get_feature_for(name).transformation,
                    coefficients=coefficients,
                    actual_vs_expected=ae_bins,
                    residual_pattern=resid_pattern,
                    significance=significance,
                    score_test=score_test,
                )
            )

        return factors

    def _get_transformation(self, name: str) -> str | None:
        """Find a representative transformation feature name for ``name``.

        Routes through the strict-matching per-variable cache so that a
        variable like ``Age`` is never mis-attributed to a feature like
        ``bs(VehAge, 1/4)``. Returns ``None`` when ``name`` is not in the
        model.
        """
        return self._get_feature_for(name).transformation

    def _is_in_model(self, name: str) -> bool:
        """Quick check whether ``name`` appears in any feature_name."""
        return bool(self._get_feature_for(name).indices)

    def _get_factor_terms(self, name: str) -> list[str]:
        """Return the list of model terms that include ``name``."""
        return list(self._get_feature_for(name).feature_names)

    def _get_factor_coefficients(
        self,
        name: str,
        result,
    ) -> list[FactorCoefficient] | None:
        """Extract coefficients for all terms involving this factor.

        Uses the strict-matching per-variable cache to avoid substring false
        positives (e.g. ``Age`` matching ``bs(VehAge, 1/4)``). The cache is
        populated lazily on first request and reused across all callers.

        B3 fix: ``relativity = exp(β)`` is suppressed for spline / expression
        / interaction terms because per-coefficient ``exp(β_k)`` of one
        component has no standalone interpretation. It is still emitted for
        single-coefficient terms (linear, target encoding, single-coef
        categorical level, and other word-boundary transforms).
        """
        if result is None or not hasattr(result, "params"):
            return None

        feat = self._get_feature_for(name)

        try:
            params = result.params
            if callable(params):
                params = params()
            if hasattr(params, "tolist"):
                params = params.tolist() if hasattr(params, "tolist") else list(params)

            feature_names = (
                result.feature_names if hasattr(result, "feature_names") else self.feature_names
            )

            bse = None
            if hasattr(result, "bse"):
                try:
                    bse = result.bse
                    if callable(bse):
                        bse = bse()
                except AttributeError:
                    bse = None
            elif hasattr(result, "std_errors"):
                bse = result.std_errors
                if callable(bse):
                    bse = bse()

            pvalues = None
            if hasattr(result, "pvalues"):
                try:
                    pvalues = result.pvalues
                    if callable(pvalues):
                        pvalues = pvalues()
                except AttributeError:
                    pvalues = None

            link = result.link if hasattr(result, "link") else None
            is_log_link = link in ("log", "Log")

            term_type = feat.term_type

            # B3 fix: relativity = exp(β) is only meaningful for SINGLE-coefficient
            # terms whose log-link transformation maps directly to a multiplicative
            # factor. For multi-coefficient or composite terms (spline / expression /
            # interaction) the per-coefficient exp(β) has no standalone meaning,
            # so we suppress it.
            relativity_meaningful = term_type in ("linear", "te", "categorical", "other")

            coefficients: list[FactorCoefficient] = []
            for i in feat.indices:
                fn = feature_names[i] if i < len(feature_names) else self.feature_names[i]
                # Preserve historical behavior: skip pure interaction features
                # in the coefficient table (they belong to the interaction
                # entry rather than to either parent factor).
                if ":" in fn:
                    continue
                coef = float(params[i])
                se = float(bse[i]) if bse is not None else 0.0
                z_val = coef / se if se > 0 else 0.0
                p_val = (
                    float(pvalues[i])
                    if pvalues is not None
                    else (2 * (1 - min(0.9999, abs(z_val) / 4)))
                )

                rel: float | None
                if is_log_link and relativity_meaningful:
                    rel = float(np.exp(coef))
                else:
                    rel = None

                coefficients.append(
                    FactorCoefficient(
                        term=fn,
                        estimate=round(coef, 6),
                        std_error=round(se, 6),
                        z_value=round(z_val, 3),
                        p_value=round(p_val, 4),
                        relativity=round(rel, 4) if rel is not None else None,
                    )
                )

            return coefficients if coefficients else None
        except (ValueError, RuntimeError, np.linalg.LinAlgError, ArithmeticError) as e:
            raise FittingError(f"Failed to extract coefficient table: {e}") from e

    def compute_factor_significance(
        self,
        name: str,
        result,
        bread_matrix: np.ndarray | None = None,
    ) -> FactorSignificance | None:
        """
        Compute significance tests for a factor in the model.

        Uses the joint Wald test β_S' @ Cov_SS⁻¹ @ β_S when the covariance
        matrix is available (essential for multi-parameter terms like splines).
        Falls back to sum of individual z² when covariance is unavailable.

        Parameter selection routes through the strict-matching per-variable
        cache so a variable like ``Age`` cannot pull in coefficients from
        ``bs(VehAge, 1/4)``.
        """
        if not hasattr(result, "params") or not hasattr(result, "bse"):
            return None

        param_indices = list(self._get_feature_for(name).indices)

        if not param_indices:
            return None

        try:
            params = np.asarray(result.params)
            bse = np.asarray(result.bse())
        except AttributeError:
            return None

        try:
            idx = np.array(param_indices)
            beta_s = params[idx]

            if bread_matrix is not None and len(idx) > 1:
                # Joint Wald test: χ² = β_S' @ Cov_SS⁻¹ @ β_S
                # Cov = scale * bread_matrix; infer scale from bse
                bread_sub = bread_matrix[np.ix_(idx, idx)]
                # scale = bse[i]² / bread[i,i] (use first valid diagonal)
                scale = 1.0
                for i in idx:
                    if bread_matrix[i, i] > 0 and bse[i] > 0:
                        scale = (bse[i] ** 2) / bread_matrix[i, i]
                        break
                cov_sub = scale * bread_sub
                cov_inv = np.linalg.inv(cov_sub)
                wald_chi2 = float(beta_s @ cov_inv @ beta_s)
            else:
                # Single parameter or no covariance: sum of individual z²
                wald_chi2 = 0.0
                for i in idx:
                    if bse[i] > 0:
                        wald_chi2 += (params[i] / bse[i]) ** 2

            df = len(param_indices)
            wald_pvalue = 1 - _chi2_cdf(wald_chi2, float(df)) if df > 0 else 1.0
            deviance_contribution = float(wald_chi2)

            return FactorSignificance(
                chi2=round(float(wald_chi2), 2),
                p=round(float(wald_pvalue), 4),
                dev_contrib=round(deviance_contribution, 2),
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError, ArithmeticError) as e:
            raise FittingError(f"Failed to compute factor significance for '{name}': {e}") from e

    def _compute_ae_continuous(self, values: np.ndarray, n_bins: int) -> list[ActualExpectedBin]:
        """Compute A/E for a continuous factor via the Rust backend."""
        rust_bins = _rust_ae_continuous(
            values,
            self.y,
            self.mu,
            self.exposure,
            n_bins,
            self.family,
            prior_weights=self._prior_weights,
            base=self.base_mu,
        )
        return self._format_ae_bins(rust_bins)

    @staticmethod
    def _format_ae_bins(rust_bins) -> list[ActualExpectedBin]:
        """Convert raw Rust A/E bins into the compact ``ActualExpectedBin`` form.

        The Rust kernel returns prior-weighted sums and, when a benchmark/base
        array was supplied, ``base_sum`` (Σ prior_weights · base prediction);
        the base-overlay rate, total, and A/E are derived from it here.
        """
        bins: list[ActualExpectedBin] = []
        for b in rust_bins:
            if b["count"] <= 0:
                continue
            exposure = b["exposure"]
            actual_sum = b["actual_sum"]
            predicted_sum = b["predicted_sum"]
            base_sum = b["base_sum"]  # None unless a base array was supplied
            base_rate = base_sum / exposure if base_sum is not None and exposure > 0 else None
            base_ae = actual_sum / base_sum if base_sum is not None and base_sum > 0.0 else None
            bins.append(
                ActualExpectedBin(
                    bin=b["bin_label"],
                    n=b["count"],
                    exposure=round(exposure, 2),
                    actual=round(actual_sum / exposure, 6) if exposure > 0 else 0.0,
                    expected=round(predicted_sum / exposure, 6) if exposure > 0 else 0.0,
                    ae_ratio=round(b["actual_expected_ratio"], 2),
                    ae_ci=[round(b["ae_ci_lower"], 2), round(b["ae_ci_upper"], 2)],
                    actual_total=float(actual_sum),
                    expected_total=float(predicted_sum),
                    base_expected=round(base_rate, 6) if base_rate is not None else None,
                    base_expected_total=float(base_sum) if base_sum is not None else None,
                    base_ae_ratio=round(base_ae, 2) if base_ae is not None else None,
                )
            )
        return bins

    def _compute_ae_categorical(
        self,
        values: np.ndarray,
        rare_threshold_pct: float,
        max_levels: int,
    ) -> list[ActualExpectedBin]:
        """Compute A/E for a categorical factor via the Rust backend."""
        levels = values.tolist()
        rust_bins = _rust_ae_categorical(
            levels,
            self.y,
            self.mu,
            self.exposure,
            rare_threshold_pct,
            max_levels,
            self.family,
            prior_weights=self._prior_weights,
            base=self.base_mu,
        )
        return self._format_ae_bins(rust_bins)

    def _compute_residual_pattern_continuous(
        self,
        values: np.ndarray,
        n_bins: int,
    ) -> ResidualPattern:
        """Compute residual pattern using Rust backend (compressed: no mean_by_bin)."""
        valid_mask = ~np.isnan(values) & ~np.isinf(values)

        if not np.any(valid_mask):
            return ResidualPattern(resid_corr=0.0, var_explained=0.0)

        result = _rust_residual_pattern(values, self.pearson_residuals, n_bins)
        return self._format_residual_pattern_continuous(result)

    @staticmethod
    def _format_residual_pattern_continuous(raw) -> ResidualPattern:
        """Convert raw Rust residual-pattern dict into the compact form."""
        corr = raw["correlation_with_residuals"]
        corr_val = float(corr) if not np.isnan(corr) else 0.0
        return ResidualPattern(
            resid_corr=round(corr_val, 4),
            var_explained=round(corr_val**2, 6),
        )

    def _compute_residual_pattern_categorical(
        self, values: np.ndarray, precomputed_unique_inverse=None
    ) -> ResidualPattern:
        """Compute residual pattern for categorical factor (compressed).

        Uses np.bincount for O(n) aggregation instead of per-level masking.
        """
        if precomputed_unique_inverse is not None:
            _unique_levels, inverse = precomputed_unique_inverse
        else:
            _unique_levels, inverse = np.unique(values, return_inverse=True)
        k = len(_unique_levels)

        level_counts = np.bincount(inverse, minlength=k).astype(np.float64)
        resid_sums = np.bincount(inverse, weights=self.pearson_residuals, minlength=k)
        level_means = np.divide(resid_sums, level_counts, out=np.zeros(k), where=level_counts > 0)

        # `overall_mean` and `ss_total` depend only on self.pearson_residuals,
        # not on the per-factor values. Cache the (mean, ss) pair on the
        # computer instance so the 6 categorical factors share the work
        # instead of each materializing an n-element `(resid - mean)**2`
        # numpy array (8 MB transient × 6 = 48 MB peak avoided).
        cache = getattr(self, "_pearson_resid_moments_cache", None)
        if cache is None or cache[0] is not self.pearson_residuals:
            resid = self.pearson_residuals
            n_resid = len(resid)
            overall_mean = float(resid.sum() / n_resid)
            # Use the algebraic identity sum((x - mean)^2) = sum(x^2) - n*mean^2
            # with np.dot for the sum-of-squares — no n-element temp allocation
            # (vs `(x - mean)**2` which materialises an 8 MB float64 array).
            sum_sq = float(np.dot(resid, resid))
            ss_total = sum_sq - n_resid * overall_mean * overall_mean
            cache = (resid, overall_mean, ss_total)
            self._pearson_resid_moments_cache = cache
        _, overall_mean, ss_total = cache

        ss_between = float(np.sum(level_counts * (level_means - overall_mean) ** 2))

        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        mean_abs_resid = np.mean(np.abs(level_means))

        return ResidualPattern(
            resid_corr=round(float(mean_abs_resid), 4),
            var_explained=round(float(eta_squared), 6),
        )

    def _compute_score_test_continuous(
        self,
        values: np.ndarray,
        design_matrix: np.ndarray,
        bread_matrix: np.ndarray,
        irls_weights: np.ndarray,
    ) -> ScoreTestResult | None:
        """Compute Rao's score test for a continuous unfitted factor.

        Returns None only when the Rust kernel reports a numerical failure
        (singular sub-matrix, non-finite quantities). Programming errors
        (TypeError, KeyError, etc.) propagate so they surface at the public
        API boundary instead of silently producing missing diagnostics.
        """
        try:
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            z = values.copy()
            if not np.all(valid_mask):
                z[~valid_mask] = np.mean(values[valid_mask]) if np.any(valid_mask) else 0.0

            result = _rust_score_test_continuous(
                z, design_matrix, self.y, self.mu, irls_weights, bread_matrix, self.family
            )

            return ScoreTestResult(
                statistic=round(result["statistic"], 2),
                df=result["df"],
                pvalue=round(result["pvalue"], 4),
                significant=result["significant"],
            )
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:
            import logging

            logging.getLogger(__name__).warning(
                "Score test computation failed for continuous factor: %s. "
                "This may indicate numerical issues with the design matrix or IRLS weights.",
                e,
            )
            return None

    def _compute_score_test_categorical(
        self,
        values: np.ndarray,
        design_matrix: np.ndarray,
        bread_matrix: np.ndarray,
        irls_weights: np.ndarray,
    ) -> ScoreTestResult | None:
        """Compute Rao's score test for a categorical unfitted factor.

        Uses target encoding (CatBoost-style): computes the mean target value
        for each level and tests this as a single continuous variable (df=1).

        Returns None only when the Rust kernel reports a numerical failure
        (singular sub-matrix, non-finite quantities). Programming errors
        propagate so they surface at the public API boundary.
        """
        try:
            unique_levels = np.unique(values)
            if len(unique_levels) < 2:
                return None

            if self.exposure is not None:
                rates = self.y / np.maximum(self.exposure, EPSILON)
            else:
                rates = self.y

            level_means = {}
            for level in unique_levels:
                mask = values == level
                if np.sum(mask) > 0:
                    level_means[level] = np.mean(rates[mask])
                else:
                    level_means[level] = np.mean(rates)

            z = np.array([level_means[v] for v in values], dtype=np.float64)

            valid_mask = np.isfinite(z)
            if not np.all(valid_mask):
                z = z.copy()
                z[~valid_mask] = np.mean(z[valid_mask]) if np.any(valid_mask) else 0.0

            result = _rust_score_test_continuous(
                z, design_matrix, self.y, self.mu, irls_weights, bread_matrix, self.family
            )

            return ScoreTestResult(
                statistic=round(result["statistic"], 2),
                df=result["df"],
                pvalue=round(result["pvalue"], 4),
                significant=result["significant"],
            )
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:
            import logging

            logging.getLogger(__name__).warning(
                "Score test computation failed for categorical factor: %s. "
                "This may indicate numerical issues with the target encoding or design matrix.",
                e,
            )
            return None

    def _linear_trend_test(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Simple linear regression trend test."""
        n = len(x)
        if n < 3:
            return float("nan"), float("nan")

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx == 0:
            return 0.0, 1.0

        slope = ss_xy / ss_xx

        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)

        df = n - 2
        mse = ss_res / df if df > 0 else 0
        se_slope = np.sqrt(mse / ss_xx) if mse > 0 and ss_xx > 0 else float("nan")

        if np.isnan(se_slope) or se_slope == 0:
            return slope, float("nan")

        t_stat = slope / se_slope
        pvalue = 2 * (1 - _t_cdf(abs(t_stat), float(df)))

        return slope, pvalue
