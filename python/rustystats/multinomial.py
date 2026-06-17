"""Native baseline-category multinomial logit API."""

from __future__ import annotations

import copy
import json
import math
import pickle
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rustystats._rustystats import (
    apply_exposure_weighted_target_encoding_py as _apply_exposure_weighted_te_rust,
)
from rustystats._rustystats import fit_multinomial_py as _fit_multinomial_rust
from rustystats._rustystats import (
    target_encode_interaction_with_exposure_py as _target_encode_interaction_with_exposure_rust,
)
from rustystats._rustystats import (
    target_encode_with_exposure_py as _target_encode_with_exposure_rust,
)
from rustystats.constants import (
    ALPHA_MAX_FLOOR,
    DEFAULT_ALPHA_MIN_RATIO,
    DEFAULT_CV_SEED,
    DEFAULT_ELASTIC_NET_L1_RATIO,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_N_ALPHAS,
    DEFAULT_N_LAMBDA,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_PRIOR_WEIGHT,
)
from rustystats.exceptions import PredictionError, ValidationError
from rustystats.formula import (
    _collect_lazyframe,
    _compute_predict_chunk_size,
    _extract_needed_columns,
    _get_constraint_indices,
    dict_to_parsed_formula,
)
from rustystats.glm import _normal_two_sided_p
from rustystats.input_transforms import (
    CompiledInputTransform,
    apply_input_transforms,
    compile_input_transforms,
    input_transform_source_columns,
    validate_input_transforms,
)
from rustystats.interactions import InteractionBuilder, ParsedFormula, TargetEncodingTermSpec

_DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_DENSE_PARAMETERS = 5000
_MIN_WEIGHTED_STD = 1e-12
_SCHEMA_VERSION = 2


def _resolve_regularization(
    alpha: float,
    l1_ratio: float,
    regularization: str | None,
) -> tuple[float, float]:
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValidationError("alpha must be finite and non-negative.")
    if not np.isfinite(l1_ratio) or l1_ratio < 0.0 or l1_ratio > 1.0:
        raise ValidationError("l1_ratio must be finite and in [0, 1].")
    if regularization is None:
        return float(alpha), float(l1_ratio if alpha > 0.0 else 0.0)
    if regularization == "ridge":
        return float(alpha), 0.0
    if regularization == "lasso":
        return float(alpha), 1.0 if alpha > 0.0 else 0.0
    if regularization == "elastic_net":
        effective_l1 = l1_ratio if l1_ratio > 0.0 else DEFAULT_ELASTIC_NET_L1_RATIO
        return float(alpha), float(effective_l1 if alpha > 0.0 else 0.0)
    raise ValidationError("regularization must be one of None, 'ridge', 'lasso', or 'elastic_net'.")


def _as_float_array(values: Any, *, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if length is not None and arr.shape[0] != length:
        raise ValidationError(f"{name} length {arr.shape[0]} does not match expected {length}.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def _as_bool_array(values: Any, *, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if length is not None and arr.shape[0] != length:
        raise ValidationError(f"{name} length {arr.shape[0]} does not match expected {length}.")
    return arr


def _unique_temp_column(columns: list[str] | set[str], base: str) -> str:
    existing = set(columns)
    name = base
    counter = 0
    while name in existing:
        counter += 1
        name = f"{base}_{counter}"
    return name


def _string_labels(values: Any) -> list[str]:
    return [str(value) for value in values]


def _derive_classes(response_series: Any) -> list[str]:
    try:
        categories = response_series.cat.get_categories()
        values = [str(value) for value in categories.to_list()]
        if values:
            return values
    except Exception:
        pass
    return sorted({str(value) for value in response_series.to_list()})


def _spec_has_array(spec: Any) -> bool:
    if spec is None:
        return False
    if isinstance(spec, np.ndarray):
        return True
    if isinstance(spec, dict):
        return any(
            isinstance(value, np.ndarray) or _array_like_not_string(value)
            for value in spec.values()
        )
    return _array_like_not_string(spec)


def _array_like_not_string(value: Any) -> bool:
    return not isinstance(value, (str, bool, np.bool_)) and hasattr(value, "__array__")


def _spec_column_names(spec: Any) -> set[str]:
    if spec is None:
        return set()
    if isinstance(spec, str):
        return {spec}
    if isinstance(spec, dict):
        return {value for value in spec.values() if isinstance(value, str)}
    return set()


def _extra_needed_columns(
    specs: list[Any],
    input_transforms: list[dict[str, Any]] | None,
) -> set[str]:
    produced = {
        spec.get("output")
        for spec in (input_transforms or [])
        if isinstance(spec, dict) and isinstance(spec.get("output"), str)
    }
    needed: set[str] = set()
    for spec in specs:
        for column in _spec_column_names(spec):
            if column not in produced:
                needed.add(column)
    return needed


def _alternative_columns_raw(alternative_terms: dict[str, Any] | None) -> set[str]:
    if alternative_terms is None:
        return set()
    if not isinstance(alternative_terms, dict):
        raise ValidationError("alternative_terms must be a dict.")
    columns: set[str] = set()
    for term_name, spec in alternative_terms.items():
        if not isinstance(spec, dict):
            raise ValidationError(f"alternative_terms[{term_name!r}] must be a dict.")
        term_columns = spec.get("columns")
        if not isinstance(term_columns, dict) or not term_columns:
            raise ValidationError(
                f"alternative_terms[{term_name!r}] must define a non-empty columns dict."
            )
        for class_label, column in term_columns.items():
            if not isinstance(column, str):
                raise ValidationError(
                    f"alternative_terms[{term_name!r}].columns[{class_label!r}] "
                    "must be a column name."
                )
            columns.add(column)
    return columns


def _alternative_needed_columns(
    alternative_terms: dict[str, Any] | None,
    input_transforms: list[dict[str, Any]] | None,
) -> set[str]:
    produced = {
        spec.get("output")
        for spec in (input_transforms or [])
        if isinstance(spec, dict) and isinstance(spec.get("output"), str)
    }
    return {
        column for column in _alternative_columns_raw(alternative_terms) if column not in produced
    }


def _normalize_alternative_terms(
    alternative_terms: dict[str, Any] | None,
    classes: list[str],
    reference: str | None = None,
) -> dict[str, dict[str, Any]]:
    if alternative_terms is None:
        return {}
    class_set = set(classes)
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, spec in alternative_terms.items():
        term_name = str(raw_name)
        if term_name in normalized:
            raise ValidationError(f"duplicate alternative term name {term_name!r}.")
        if not isinstance(spec, dict):
            raise ValidationError(f"alternative_terms[{term_name!r}] must be a dict.")
        coefficient = str(spec.get("coefficient", "generic")).replace("-", "_")
        if coefficient not in {"generic", "class_specific"}:
            raise ValidationError(
                f"alternative_terms[{term_name!r}].coefficient must be "
                "'generic' or 'class_specific'."
            )
        transform = str(spec.get("transform", "identity")).lower()
        if transform not in {"identity", "log"}:
            raise ValidationError(
                f"alternative_terms[{term_name!r}].transform must be 'identity' or 'log'."
            )
        columns: dict[str, str] = {}
        for class_label, column in spec["columns"].items():
            label = str(class_label)
            if label not in class_set:
                raise ValidationError(
                    f"alternative_terms[{term_name!r}] references unknown class {label!r}."
                )
            if (
                coefficient == "class_specific"
                and reference is not None
                and label == str(reference)
            ):
                raise ValidationError(
                    f"alternative_terms[{term_name!r}] is class_specific but defines a column "
                    f"for the reference class {label!r}. Class-specific alternative coefficients "
                    "are baseline-relative, so the reference column would be silently ignored. "
                    "Remove it, or use coefficient='generic' (which legitimately uses the "
                    "reference alternative)."
                )
            columns[label] = column
        normalized[term_name] = {
            "columns": columns,
            "coefficient": coefficient,
            "transform": transform,
        }
    return normalized


def _alternative_names_by_kind(
    alternative_terms: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    generic = [name for name, spec in alternative_terms.items() if spec["coefficient"] == "generic"]
    specific = [
        name for name, spec in alternative_terms.items() if spec["coefficient"] == "class_specific"
    ]
    return generic, specific


def _apply_alternative_transform(
    values: np.ndarray, *, term_name: str, transform: str
) -> np.ndarray:
    if transform == "identity":
        return values
    if transform == "log":
        if np.any(values <= 0.0):
            raise ValidationError(
                f"alternative term {term_name!r} uses transform='log' but contains "
                "non-positive values."
            )
        return np.log(values)
    raise AssertionError(f"unsupported alternative transform {transform!r}")


def _resolve_alternative_arrays(
    data: Any,
    classes: list[str],
    alternative_terms: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    generic_names, specific_names = _alternative_names_by_kind(alternative_terms)
    generic = np.zeros((len(data), len(classes), len(generic_names)), dtype=np.float64)
    specific = np.zeros((len(data), len(classes), len(specific_names)), dtype=np.float64)

    generic_pos = {name: idx for idx, name in enumerate(generic_names)}
    specific_pos = {name: idx for idx, name in enumerate(specific_names)}
    for term_name, spec in alternative_terms.items():
        target = generic if spec["coefficient"] == "generic" else specific
        term_idx = (
            generic_pos[term_name] if spec["coefficient"] == "generic" else specific_pos[term_name]
        )
        for class_idx, class_label in enumerate(classes):
            column = spec["columns"].get(class_label)
            if column is None:
                continue
            if column not in data.columns:
                raise ValidationError(f"alternative term {term_name!r} requires column {column!r}.")
            values = _as_float_array(
                data[column].to_numpy(), name=f"alternative_terms[{term_name!r}]", length=len(data)
            )
            target[:, class_idx, term_idx] = _apply_alternative_transform(
                values, term_name=term_name, transform=spec["transform"]
            )

    return generic, specific, generic_names, specific_names


def _parsed_without_target_encoding(parsed: ParsedFormula) -> ParsedFormula:
    """Return a shared-X parsed formula with multinomial TE terms removed."""
    stripped = copy.copy(parsed)
    stripped.target_encoding_terms = []
    stripped._te_by_var = None
    return stripped


def _target_encoding_feature_name(term: TargetEncodingTermSpec) -> str:
    if term.interaction_vars is not None and len(term.interaction_vars) >= 2:
        return f"TE({':'.join(term.interaction_vars)})"
    return f"TE({term.var_name})"


def _target_encoding_source_columns(
    terms: dict[str, dict[str, Any]],
    interactions: list[dict[str, Any]] | None,
    input_transforms: list[dict[str, Any]] | None,
) -> set[str]:
    transform_outputs = {
        spec.get("output")
        for spec in (input_transforms or [])
        if isinstance(spec, dict) and isinstance(spec.get("output"), str)
    }
    columns: set[str] = set()
    for var_name, spec in terms.items():
        if spec.get("type") == "target_encoding":
            column = str(spec.get("variable", var_name))
            if column not in transform_outputs:
                columns.add(column)
    reserved = {
        "include_main",
        "target_encoding",
        "frequency_encoding",
        "prior_weight",
        "n_permutations",
        "mode",
    }
    for interaction in interactions or []:
        if not interaction.get("target_encoding"):
            continue
        for var_name in interaction:
            if var_name in reserved:
                continue
            if var_name not in transform_outputs:
                columns.add(var_name)
    return columns


def _target_encoding_categories(data: Any, term: TargetEncodingTermSpec) -> list[str]:
    import polars as pl

    if term.interaction_vars is not None and len(term.interaction_vars) >= 2:
        missing = [var for var in term.interaction_vars if var not in data.columns]
        if missing:
            raise PredictionError(
                f"target encoding interaction requires missing columns {missing}."
            )
        return data.select(
            pl.concat_str(
                [pl.col(var).cast(pl.Utf8) for var in term.interaction_vars],
                separator=":",
            ).alias("__rustystats_te_key__")
        )["__rustystats_te_key__"].to_list()
    if term.var_name not in data.columns:
        raise PredictionError(f"target encoding requires column {term.var_name!r}.")
    return data[term.var_name].cast(pl.Utf8).to_list()


@dataclass
class _MultinomialTargetEncodingTermState:
    feature_name: str
    var_name: str
    interaction_vars: list[str] | None
    prior_weight: float
    n_permutations: int
    class_stats: dict[str, dict[str, Any]]

    def transform(self, data: Any, classes: list[str]) -> np.ndarray:
        categories = _target_encoding_categories(
            data,
            TargetEncodingTermSpec(
                var_name=self.var_name,
                prior_weight=self.prior_weight,
                n_permutations=self.n_permutations,
                interaction_vars=None
                if self.interaction_vars is None
                else list(self.interaction_vars),
            ),
        )
        encoded = np.zeros((len(data), len(classes)), dtype=np.float64)
        for class_idx, class_label in enumerate(classes):
            state = self.class_stats.get(class_label)
            if state is None:
                continue
            encoded[:, class_idx] = np.asarray(
                _apply_exposure_weighted_te_rust(
                    categories,
                    state["stats"],
                    float(state["prior"]),
                    self.prior_weight,
                ),
                dtype=np.float64,
            )
        return encoded


@dataclass
class _MultinomialTargetEncodingState:
    classes: list[str]
    reference: str
    non_reference_classes: list[str]
    mode: str
    fallback: str
    terms: list[_MultinomialTargetEncodingTermState]

    @property
    def feature_names(self) -> list[str]:
        return [term.feature_name for term in self.terms]

    def transform(self, data: Any) -> np.ndarray:
        tensor = np.zeros((len(data), len(self.classes), len(self.terms)), dtype=np.float64)
        for term_idx, term in enumerate(self.terms):
            tensor[:, :, term_idx] = term.transform(data, self.classes)
        return tensor


@dataclass
class MultinomialFoldPreprocessingState:
    """Fold-local preprocessing state for future multinomial CV paths."""

    builder: InteractionBuilder
    parsed_formula: ParsedFormula
    target_encoding_state: _MultinomialTargetEncodingState | None
    alternative_generic_feature_names: list[str]
    alternative_specific_feature_names: list[str]


@dataclass
class MultinomialFoldDesign:
    """All solver inputs for one fold-safe multinomial preprocessing split."""

    x_train: np.ndarray
    x_val: np.ndarray
    alternative_generic_train: np.ndarray
    alternative_generic_val: np.ndarray
    alternative_specific_train: np.ndarray
    alternative_specific_val: np.ndarray
    availability_train: np.ndarray
    availability_val: np.ndarray
    offset_train: np.ndarray
    offset_val: np.ndarray
    weights_train: np.ndarray | None
    weights_val: np.ndarray | None
    y_train: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    preprocessing_state: MultinomialFoldPreprocessingState


@dataclass
class MultinomialSmoothTermResult:
    """Diagnostics for one automatically-smoothed shared multinomial term."""

    variable: str
    spline_type: str
    k: int
    lambda_: float
    edf: float
    gcv: float
    col_start: int
    col_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "type": self.spline_type,
            "k": self.k,
            "lambda": self.lambda_,
            "edf": self.edf,
            "gcv": self.gcv,
            "col_start": self.col_start,
            "col_end": self.col_end,
        }


def _fold_local_parsed_formula(parsed: ParsedFormula) -> ParsedFormula:
    from rustystats.regularization_path import reset_fold_local_spline_state

    parsed_fold = copy.deepcopy(parsed)
    reset_fold_local_spline_state(parsed_fold)
    return parsed_fold


def _fit_target_encoding_term_for_class(
    data: Any,
    term: TargetEncodingTermSpec,
    claims: np.ndarray,
    exposure: np.ndarray,
    seed: int | None,
) -> tuple[np.ndarray, float, dict[str, tuple[float, float]]]:
    if term.interaction_vars is not None and len(term.interaction_vars) == 2:
        import polars as pl

        var1, var2 = term.interaction_vars
        encoded, _name, prior, stats = _target_encode_interaction_with_exposure_rust(
            data[var1].cast(pl.Utf8).to_list(),
            data[var2].cast(pl.Utf8).to_list(),
            claims,
            exposure,
            var1,
            var2,
            term.prior_weight,
            term.n_permutations,
            seed,
        )
    else:
        categories = _target_encoding_categories(data, term)
        encoded, _name, prior, stats = _target_encode_with_exposure_rust(
            categories,
            claims,
            exposure,
            _target_encoding_feature_name(term)[3:-1],
            term.prior_weight,
            term.n_permutations,
            seed,
        )
    return np.asarray(encoded, dtype=np.float64), float(prior), stats


def _build_multinomial_target_encoding(
    data: Any,
    terms: list[TargetEncodingTermSpec],
    *,
    classes: list[str],
    reference: str,
    y_codes: np.ndarray,
    availability: np.ndarray,
    row_weights: np.ndarray | None,
    seed: int | None,
) -> tuple[np.ndarray, _MultinomialTargetEncodingState | None]:
    if not terms:
        return np.zeros((len(data), len(classes), 0), dtype=np.float64), None

    weights = (
        np.ones(len(data), dtype=np.float64)
        if row_weights is None
        else np.asarray(row_weights, dtype=np.float64)
    )
    tensor = np.zeros((len(data), len(classes), len(terms)), dtype=np.float64)
    non_reference = [label for label in classes if label != reference]
    term_states: list[_MultinomialTargetEncodingTermState] = []
    for term_idx, term in enumerate(terms):
        feature_name = _target_encoding_feature_name(term)
        class_stats: dict[str, dict[str, Any]] = {}
        for class_idx, class_label in enumerate(classes):
            if class_label == reference:
                continue
            claims = weights * (y_codes == class_idx).astype(np.float64)
            exposure = weights * availability[:, class_idx].astype(np.float64)
            encoded, prior, stats = _fit_target_encoding_term_for_class(
                data,
                term,
                claims.astype(np.float64, copy=False),
                exposure.astype(np.float64, copy=False),
                seed,
            )
            tensor[:, class_idx, term_idx] = encoded
            class_stats[class_label] = {
                "prior": prior,
                "stats": stats,
            }
        term_states.append(
            _MultinomialTargetEncodingTermState(
                feature_name=feature_name,
                var_name=term.var_name,
                interaction_vars=None
                if term.interaction_vars is None
                else list(term.interaction_vars),
                prior_weight=float(term.prior_weight),
                n_permutations=int(term.n_permutations),
                class_stats=class_stats,
            )
        )

    return tensor, _MultinomialTargetEncodingState(
        classes=list(classes),
        reference=reference,
        non_reference_classes=non_reference,
        mode="alternative_specific_diagonal",
        fallback="prior",
        terms=term_states,
    )


def _append_alternative_specific_terms(
    existing: np.ndarray,
    extra: np.ndarray,
) -> np.ndarray:
    if extra.shape[2] == 0:
        return existing
    if existing.shape[2] == 0:
        return extra
    return np.concatenate([existing, extra], axis=2)


def _weighted_standardization_scale(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return 1.0
    mean = float(weights @ values / weight_sum)
    centered = values - mean
    variance = float(weights @ (centered * centered) / weight_sum)
    scale = math.sqrt(max(variance, 0.0))
    if not np.isfinite(scale) or scale <= _MIN_WEIGHTED_STD:
        return 1.0
    return scale


def _alternative_standardization(
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    availability: np.ndarray,
    weights: np.ndarray | None,
    reference_index: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Scale-only standardization metadata for penalized alternative terms."""
    row_weights = (
        np.ones(alternative_generic.shape[0], dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    n_generic = alternative_generic.shape[2]
    generic_center = generic_scale = None
    if n_generic:
        generic_center = np.zeros(n_generic, dtype=np.float64)
        generic_scale = np.ones(n_generic, dtype=np.float64)
        cell_weights = row_weights[:, None] * availability.astype(np.float64, copy=False)
        active = availability
        for term_idx in range(n_generic):
            generic_scale[term_idx] = _weighted_standardization_scale(
                alternative_generic[:, :, term_idx][active],
                cell_weights[active],
            )

    n_specific = alternative_specific.shape[2]
    specific_center = specific_scale = None
    if n_specific:
        non_reference = [
            class_idx
            for class_idx in range(alternative_specific.shape[1])
            if class_idx != reference_index
        ]
        specific_center = np.zeros((len(non_reference), n_specific), dtype=np.float64)
        specific_scale = np.ones((len(non_reference), n_specific), dtype=np.float64)
        for block_idx, class_idx in enumerate(non_reference):
            active = availability[:, class_idx]
            for term_idx in range(n_specific):
                specific_scale[block_idx, term_idx] = _weighted_standardization_scale(
                    alternative_specific[:, class_idx, term_idx][active],
                    row_weights[active],
                )

    return generic_center, generic_scale, specific_center, specific_scale


def _multinomial_standardization_metadata(
    x: np.ndarray,
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    availability: np.ndarray,
    weights: np.ndarray | None,
    reference_index: int,
    *,
    fit_intercept: bool,
    standardize: bool,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if not standardize:
        return None, None, None, None, None, None

    from rustystats.regularization_path import compute_standardization, solver_standardization

    center, scale = compute_standardization(
        x,
        weights,
        fit_intercept=fit_intercept,
    )
    center, scale = solver_standardization(center, scale, fit_intercept=fit_intercept)
    (
        generic_center,
        generic_scale,
        specific_center,
        specific_scale,
    ) = _alternative_standardization(
        alternative_generic,
        alternative_specific,
        availability,
        weights,
        reference_index,
    )
    return center, scale, generic_center, generic_scale, specific_center, specific_scale


def _initial_theta_from_multinomial_result(
    result: Any | None,
    *,
    center: np.ndarray | None,
    scale: np.ndarray | None,
    alternative_generic_center: np.ndarray | None,
    alternative_generic_scale: np.ndarray | None,
    alternative_specific_center: np.ndarray | None,
    alternative_specific_scale: np.ndarray | None,
    fit_intercept: bool,
) -> np.ndarray | None:
    if result is None:
        return None

    params_original = np.asarray(result.params, dtype=np.float64)
    params_solver = params_original.copy()
    generic_solver = np.asarray(result.alternative_generic_coefficients, dtype=np.float64).copy()
    specific_solver = np.asarray(result.alternative_specific_coefficients, dtype=np.float64).copy()

    if alternative_generic_scale is not None and generic_solver.size:
        generic_solver *= np.asarray(alternative_generic_scale, dtype=np.float64)
    if alternative_specific_scale is not None and specific_solver.size:
        specific_solver *= np.asarray(alternative_specific_scale, dtype=np.float64)

    if center is not None and scale is not None and params_solver.size:
        center_arr = np.asarray(center, dtype=np.float64)
        scale_arr = np.asarray(scale, dtype=np.float64)
        start_feature = 1 if fit_intercept and params_solver.shape[1] > 0 else 0
        params_solver[:, start_feature:] = (
            params_original[:, start_feature:] * scale_arr[start_feature:]
        )

    if fit_intercept and params_solver.shape[1] > 0:
        intercept_adjustment = np.zeros(params_solver.shape[0], dtype=np.float64)
        if center is not None and scale is not None and params_solver.shape[1] > 1:
            center_arr = np.asarray(center, dtype=np.float64)
            scale_arr = np.asarray(scale, dtype=np.float64)
            intercept_adjustment += np.sum(
                params_solver[:, 1:] * (-center_arr[1:] / scale_arr[1:]),
                axis=1,
            )
        if (
            alternative_specific_center is not None
            and alternative_specific_scale is not None
            and specific_solver.size
        ):
            specific_center = np.asarray(alternative_specific_center, dtype=np.float64)
            specific_scale = np.asarray(alternative_specific_scale, dtype=np.float64)
            intercept_adjustment += np.sum(
                specific_solver * (-specific_center / specific_scale),
                axis=1,
            )
        params_solver[:, 0] = params_original[:, 0] - intercept_adjustment

    _ = alternative_generic_center  # Generic centering adds a utility constant that cancels.
    initial_theta = np.concatenate(
        [
            params_solver.ravel(),
            generic_solver.ravel(),
            specific_solver.ravel(),
        ]
    ).astype(np.float64, copy=False)
    if np.any(~np.isfinite(initial_theta)):
        raise ValidationError("multinomial warm-start coefficients must be finite.")
    return np.ascontiguousarray(initial_theta, dtype=np.float64)


def _standardized_multinomial_design_for_penalty(
    x: np.ndarray,
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    availability: np.ndarray,
    weights: np.ndarray | None,
    reference_index: int,
    *,
    fit_intercept: bool,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_work = np.asarray(x, dtype=np.float64)
    generic_work = np.asarray(alternative_generic, dtype=np.float64).copy()
    specific_work = np.asarray(alternative_specific, dtype=np.float64).copy()
    (
        center,
        scale,
        generic_center,
        generic_scale,
        specific_center,
        specific_scale,
    ) = _multinomial_standardization_metadata(
        x_work,
        generic_work,
        specific_work,
        availability,
        weights,
        reference_index,
        fit_intercept=fit_intercept,
        standardize=standardize,
    )
    if center is not None and scale is not None:
        x_work = (x_work - center[None, :]) / scale[None, :]
    if generic_center is not None and generic_scale is not None:
        generic_work = (generic_work - generic_center[None, None, :]) / generic_scale[None, None, :]
    if specific_center is not None and specific_scale is not None:
        block = 0
        for class_idx in range(specific_work.shape[1]):
            if class_idx == reference_index:
                continue
            specific_work[:, class_idx, :] = (
                specific_work[:, class_idx, :] - specific_center[block][None, :]
            ) / specific_scale[block][None, :]
            block += 1
    return x_work, generic_work, specific_work


def _multinomial_null_probabilities(
    y_codes: np.ndarray,
    n_classes: int,
    reference_index: int,
    availability: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None,
    *,
    fit_intercept: bool,
) -> np.ndarray:
    if fit_intercept:
        x_null = np.ones((len(y_codes), 1), dtype=np.float64)
        null_result = _fit_multinomial_rust(
            y_codes,
            x_null,
            n_classes,
            reference_index,
            availability,
            offset,
            weights,
            0.0,
            0.0,
            100,
            1e-8,
            True,
            None,
            None,
            True,
            _DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES,
            max(_DEFAULT_MAX_DENSE_PARAMETERS, n_classes - 1),
            False,
            False,
        )
        return np.asarray(null_result.fitted_probabilities, dtype=np.float64)

    logits = np.asarray(offset, dtype=np.float64)
    masked = np.where(availability, logits, -np.inf)
    max_eta = np.max(masked, axis=1, keepdims=True)
    exp_eta = np.where(availability, np.exp(masked - max_eta), 0.0)
    denom = exp_eta.sum(axis=1, keepdims=True)
    return exp_eta / denom


def _multinomial_alpha_max_from_arrays(
    y_codes: np.ndarray,
    x: np.ndarray,
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    availability: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None,
    n_classes: int,
    reference_index: int,
    l1_ratio: float = 1.0,
    *,
    fit_intercept: bool,
    standardize: bool = True,
) -> float:
    if not np.isfinite(l1_ratio) or l1_ratio < 0.0 or l1_ratio > 1.0:
        raise ValidationError("l1_ratio must be finite and in [0, 1].")

    x_work, generic_work, specific_work = _standardized_multinomial_design_for_penalty(
        x,
        alternative_generic,
        alternative_specific,
        availability,
        weights,
        reference_index,
        fit_intercept=fit_intercept,
        standardize=standardize,
    )
    weights = (
        np.ones(len(y_codes), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    penalized_scores: list[np.ndarray] = []
    probabilities = _multinomial_null_probabilities(
        y_codes,
        n_classes,
        reference_index,
        availability,
        offset,
        weights,
        fit_intercept=fit_intercept,
    )
    for class_idx in range(n_classes):
        if class_idx == reference_index:
            continue
        residual = weights * (probabilities[:, class_idx] - (y_codes == class_idx))
        shared_scores = x_work.T @ residual
        if fit_intercept and shared_scores.size:
            shared_scores = shared_scores[1:]
        penalized_scores.append(shared_scores)
        if specific_work.shape[2]:
            penalized_scores.append(specific_work[:, class_idx, :].T @ residual)

    if generic_work.shape[2]:
        expected = np.sum(probabilities[:, :, None] * generic_work, axis=1)
        observed = generic_work[np.arange(len(y_codes)), y_codes, :]
        penalized_scores.append(np.sum(weights[:, None] * (expected - observed), axis=0))

    nonempty_scores = [
        np.asarray(scores, dtype=np.float64) for scores in penalized_scores if scores.size
    ]
    if not nonempty_scores:
        return ALPHA_MAX_FLOOR

    if l1_ratio > 0.0:
        max_score = max(float(np.max(np.abs(scores))) for scores in nonempty_scores)
        return max(max_score / l1_ratio, ALPHA_MAX_FLOOR)

    weighted_diags: list[np.ndarray] = []
    if x_work.shape[1] > 0:
        shared = x_work[:, 1:] if fit_intercept else x_work
        if shared.size:
            weighted_diags.append(np.sum((shared**2) * weights[:, None], axis=0))
    if generic_work.shape[2]:
        cell_weights = weights[:, None] * availability.astype(np.float64, copy=False)
        weighted_diags.append(np.sum((generic_work**2) * cell_weights[:, :, None], axis=(0, 1)))
    if specific_work.shape[2]:
        for class_idx in range(n_classes):
            if class_idx == reference_index:
                continue
            active_weights = weights * availability[:, class_idx].astype(np.float64)
            weighted_diags.append(
                np.sum((specific_work[:, class_idx, :] ** 2) * active_weights[:, None], axis=0)
            )
    nonempty_diags = [diag for diag in weighted_diags if diag.size]
    if not nonempty_diags:
        return ALPHA_MAX_FLOOR
    diag_values = np.concatenate(nonempty_diags)
    return max(float(np.median(diag_values)) * 10.0, ALPHA_MAX_FLOOR)


def multinomial_alpha_max(
    model: MultinomialDict,
    l1_ratio: float = 1.0,
    *,
    standardize: bool = True,
) -> float:
    """Return an alpha-grid anchor for a fitted-design multinomial dict model.

    For ``l1_ratio > 0`` this is the KKT value that zeroes all penalized
    non-intercept coefficients in the standardized coordinate system used by
    :meth:`MultinomialDict.fit`. For pure ridge, no finite all-zero KKT value
    exists, so the helper returns the same style of weighted Gram heuristic used
    by the scalar GLM regularization path.
    """
    if not isinstance(model, MultinomialDict):
        raise ValidationError("multinomial_alpha_max expects a MultinomialDict instance.")
    return _multinomial_alpha_max_from_arrays(
        model.y_codes,
        model.X,
        model.alternative_generic,
        model.alternative_specific,
        model.availability,
        model.offset,
        model.weights,
        len(model.classes_),
        model.reference_index_,
        l1_ratio,
        fit_intercept=model.intercept,
        standardize=standardize,
    )


def _multinomial_parameter_count(
    n_shared: int,
    n_classes: int,
    n_alt_generic: int,
    n_alt_specific: int,
) -> int:
    return n_shared * (n_classes - 1) + n_alt_generic + n_alt_specific * (n_classes - 1)


def _validate_multinomial_dense_fit_size(
    *,
    n_shared: int,
    n_classes: int,
    n_alt_generic: int,
    n_alt_specific: int,
    hessian_memory_limit_bytes: int,
    max_dense_parameters: int,
    context: str,
) -> None:
    q = _multinomial_parameter_count(n_shared, n_classes, n_alt_generic, n_alt_specific)
    if q > max_dense_parameters:
        raise ValidationError(
            f"multinomial {context} would estimate q={q} parameters, exceeding "
            f"max_dense_parameters={max_dense_parameters}."
        )
    hessian_bytes = q * q * 8
    if hessian_bytes > hessian_memory_limit_bytes:
        estimated_mb = hessian_bytes / (1024.0 * 1024.0)
        limit_mb = hessian_memory_limit_bytes / (1024.0 * 1024.0)
        raise ValidationError(
            f"multinomial {context} dense Hessian would require {estimated_mb:.1f} MB "
            f"for q={q} parameters, exceeding the configured {limit_mb:.1f} MB limit."
        )


def _stratified_multinomial_cv_folds(
    y_codes: np.ndarray,
    weights: np.ndarray | None,
    n_classes: int,
    cv: int,
    seed: int | None,
    *,
    max_attempts: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cv < 2:
        raise ValidationError("cv must be at least 2.")
    n = len(y_codes)
    if cv > n:
        raise ValidationError(f"cv={cv} cannot exceed the number of rows ({n}).")
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValidationError("CV weights must be finite and non-negative.")
    for class_idx in range(n_classes):
        if float(np.sum(w[y_codes == class_idx])) <= 0.0:
            raise ValidationError(
                f"class {class_idx} has zero positive effective weight; CV cannot proceed."
            )

    base_seed = DEFAULT_CV_SEED if seed is None else int(seed)
    all_indices = np.arange(n, dtype=np.int64)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(base_seed + attempt)
        fold_bins = [[] for _ in range(cv)]
        for class_idx in range(n_classes):
            class_indices = np.flatnonzero(y_codes == class_idx)
            rng.shuffle(class_indices)
            fold_offset = int(rng.integers(0, cv))
            for position, row_idx in enumerate(class_indices):
                fold_bins[(position + fold_offset) % cv].append(int(row_idx))

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        valid = True
        for fold_rows in fold_bins:
            if not fold_rows:
                valid = False
                break
            val_idx = np.asarray(sorted(fold_rows), dtype=np.int64)
            val_mask = np.zeros(n, dtype=bool)
            val_mask[val_idx] = True
            train_idx = all_indices[~val_mask]
            for class_idx in range(n_classes):
                if float(np.sum(w[train_idx][y_codes[train_idx] == class_idx])) <= 0.0:
                    valid = False
                    break
            if not valid:
                break
            folds.append((train_idx, val_idx))
        if valid and len(folds) == cv:
            return folds

    raise ValidationError(
        "could not construct CV folds whose training splits contain every observed class "
        f"with positive effective weight after {max_attempts} deterministic attempts. "
        "Reduce cv or combine rare classes."
    )


def _multinomial_probabilities_from_result(
    result: Any,
    x: np.ndarray,
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    availability: np.ndarray,
    offset: np.ndarray,
    n_classes: int,
    reference_index: int,
) -> np.ndarray:
    logits = np.asarray(offset, dtype=np.float64).copy()
    generic_coef = np.asarray(result.alternative_generic_coefficients, dtype=np.float64)
    if generic_coef.size:
        logits += np.tensordot(alternative_generic, generic_coef, axes=([2], [0]))
    params = np.asarray(result.params, dtype=np.float64)
    specific_coef = np.asarray(result.alternative_specific_coefficients, dtype=np.float64)
    block = 0
    for class_idx in range(n_classes):
        if class_idx == reference_index:
            continue
        logits[:, class_idx] += x @ params[block, :]
        if specific_coef.size:
            logits[:, class_idx] += alternative_specific[:, class_idx, :] @ specific_coef[block, :]
        block += 1
    return _masked_softmax(logits, availability)


def _multinomial_validation_deviance(
    probabilities: np.ndarray,
    y_codes: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    w = np.ones(len(y_codes), dtype=np.float64) if weights is None else np.asarray(weights)
    denom = float(np.sum(w))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("inf")
    selected = np.maximum(probabilities[np.arange(len(y_codes)), y_codes], 1e-300)
    return float(-2.0 * np.sum(w * np.log(selected)) / denom)


def _multinomial_bound_indices(
    feature_names: list[str],
    *,
    n_classes: int,
    reference_index: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    shared_nonneg, shared_nonpos = _get_constraint_indices(feature_names)
    p = len(feature_names)

    def expand(shared_indices: list[int]) -> list[int]:
        expanded: list[int] = []
        block = 0
        for class_idx in range(n_classes):
            if class_idx == reference_index:
                continue
            block_offset = block * p
            expanded.extend(block_offset + int(idx) for idx in shared_indices)
            block += 1
        return expanded

    return expand(shared_nonneg), expand(shared_nonpos), list(shared_nonneg), list(shared_nonpos)


def _multinomial_constraint_metadata(
    feature_names: list[str],
    classes: list[str],
    reference: str,
    shared_nonneg: list[int],
    shared_nonpos: list[int],
) -> list[dict[str, Any]]:
    p = len(feature_names)
    non_reference_classes = [label for label in classes if label != reference]
    sign_by_shared_index = {int(idx): "nonnegative" for idx in shared_nonneg} | {
        int(idx): "nonpositive" for idx in shared_nonpos
    }
    records = []
    for shared_idx in sorted(sign_by_shared_index):
        records.append(
            {
                "coefficient_type": "shared",
                "feature": feature_names[shared_idx],
                "shared_column_index": int(shared_idx),
                "sign": sign_by_shared_index[shared_idx],
                "classes": list(non_reference_classes),
                "parameter_indices": [
                    int(block * p + shared_idx) for block in range(len(non_reference_classes))
                ],
                "target": "class_utility_vs_reference",
                "probability_monotonicity": False,
            }
        )
    return records


def _fit_multinomial_arrays(
    *,
    y_codes: np.ndarray,
    x: np.ndarray,
    n_classes: int,
    reference_index: int,
    availability: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    fit_intercept: bool,
    standardize: bool,
    compute_covariance: bool,
    store_design_matrix: bool,
    verbose: bool,
    hessian_memory_limit_bytes: int,
    max_dense_parameters: int,
    alternative_generic: np.ndarray,
    alternative_specific: np.ndarray,
    initial_result: Any | None = None,
    smooth_col_ranges: list[tuple[int, int]] | None = None,
    smooth_penalties: list[np.ndarray] | None = None,
    smooth_lambdas: list[float] | np.ndarray | None = None,
    bound_nonneg_indices: list[int] | None = None,
    bound_nonpos_indices: list[int] | None = None,
) -> Any:
    (
        center,
        scale,
        alternative_generic_center,
        alternative_generic_scale,
        alternative_specific_center,
        alternative_specific_scale,
    ) = _multinomial_standardization_metadata(
        x,
        alternative_generic,
        alternative_specific,
        availability,
        weights,
        reference_index,
        fit_intercept=fit_intercept,
        standardize=alpha > 0.0 and standardize,
    )
    l1_active = alpha > 0.0 and l1_ratio > 0.0
    initial_theta = _initial_theta_from_multinomial_result(
        initial_result,
        center=center,
        scale=scale,
        alternative_generic_center=alternative_generic_center,
        alternative_generic_scale=alternative_generic_scale,
        alternative_specific_center=alternative_specific_center,
        alternative_specific_scale=alternative_specific_scale,
        fit_intercept=fit_intercept,
    )
    return _fit_multinomial_rust(
        y_codes,
        np.ascontiguousarray(x, dtype=np.float64),
        n_classes,
        reference_index,
        availability,
        offset,
        weights,
        alpha,
        0.0 if alpha == 0.0 else l1_ratio,
        max_iter,
        tol,
        fit_intercept,
        center,
        scale,
        not (compute_covariance and not l1_active),
        int(hessian_memory_limit_bytes),
        int(max_dense_parameters),
        store_design_matrix,
        verbose,
        alternative_generic,
        alternative_specific,
        alternative_generic_center,
        alternative_generic_scale,
        alternative_specific_center,
        alternative_specific_scale,
        initial_theta,
        smooth_col_ranges,
        smooth_penalties,
        None if smooth_lambdas is None else list(map(float, smooth_lambdas)),
        bound_nonneg_indices,
        bound_nonpos_indices,
    )


def _weighted_multinomial_gcv(deviance: float, weight_sum: float, total_edf: float | None) -> float:
    if total_edf is None or not np.isfinite(total_edf):
        return float("inf")
    if weight_sum <= 0.0 or not np.isfinite(weight_sum) or not np.isfinite(deviance):
        return float("inf")
    denominator = weight_sum - float(total_edf)
    if denominator <= 1e-8 * max(1.0, weight_sum):
        return float("inf")
    return float(deviance * weight_sum / (denominator * denominator))


def _multinomial_smooth_penalty_inputs(
    model: MultinomialDict,
) -> tuple[list[Any], list[tuple[int, int]], list[np.ndarray]]:
    smooth_terms, smooth_col_ranges = model._builder.get_smooth_terms()
    smooth_terms = list(smooth_terms)
    smooth_col_ranges = [(int(start), int(end)) for start, end in smooth_col_ranges]
    if len(smooth_terms) != len(smooth_col_ranges):
        raise ValidationError(
            "multinomial smooth metadata is inconsistent: the number of smooth terms "
            "does not match the number of smooth column ranges."
        )
    penalties: list[np.ndarray] = []
    for idx, term in enumerate(smooth_terms):
        if (
            getattr(term, "monotonicity", None) is not None
            or getattr(term, "spline_type", None) == "ms"
        ):
            raise ValidationError(
                "monotonic smooth terms are not yet supported for multinomial_dict."
            )
        start, end = smooth_col_ranges[idx]
        k = end - start
        if k <= 0:
            raise ValidationError(
                f"multinomial smooth term {getattr(term, 'var_name', idx)!r} has an empty "
                "basis column range."
            )
        penalties.append(np.ascontiguousarray(term.compute_penalty_matrix(k)[:k, :k]))
    return smooth_terms, smooth_col_ranges, penalties


def _fit_multinomial_smooth_path(
    model: MultinomialDict,
    *,
    max_iter: int,
    tol: float,
    compute_covariance: bool,
    store_design_matrix: bool,
    verbose: bool,
    hessian_memory_limit_bytes: int,
    max_dense_parameters: int,
    n_lambda: int,
    lambda_min: float,
    lambda_max: float,
    max_lambda_iter: int,
) -> tuple[Any, list[MultinomialSmoothTermResult], float, float, dict[str, Any]]:
    smooth_terms, smooth_col_ranges, smooth_penalties = _multinomial_smooth_penalty_inputs(model)
    if not smooth_terms:
        raise ValidationError("_fit_multinomial_smooth_path requires at least one smooth term.")
    if n_lambda <= 0:
        raise ValidationError("n_lambda must be positive for multinomial smooth tuning.")
    if (
        lambda_min <= 0.0
        or lambda_max <= lambda_min
        or not np.isfinite(lambda_min)
        or not np.isfinite(lambda_max)
    ):
        raise ValidationError(
            "lambda_min/lambda_max must be finite with 0 < lambda_min < lambda_max."
        )
    if max_lambda_iter <= 0:
        raise ValidationError("max_lambda_iter must be positive.")

    candidate_lambdas = np.exp(
        np.linspace(math.log(lambda_min), math.log(lambda_max), int(n_lambda))
    )
    lambdas = np.ones(len(smooth_terms), dtype=np.float64)
    weight_sum = (
        float(model.n_obs)
        if model.weights is None
        else float(np.sum(np.asarray(model.weights, dtype=np.float64)))
    )
    best_result = None
    best_gcv = float("inf")
    candidate_fit_count = 0
    converged = False

    def fit_candidate(
        trial_lambdas: np.ndarray,
        initial_result: Any | None,
    ) -> tuple[Any, float]:
        result = _fit_multinomial_arrays(
            y_codes=model.y_codes,
            x=model.X,
            n_classes=len(model.classes_),
            reference_index=model.reference_index_,
            availability=model.availability,
            offset=model.offset,
            weights=model.weights,
            alpha=0.0,
            l1_ratio=0.0,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=model.intercept,
            standardize=False,
            compute_covariance=False,
            store_design_matrix=False,
            verbose=False,
            hessian_memory_limit_bytes=hessian_memory_limit_bytes,
            max_dense_parameters=max_dense_parameters,
            alternative_generic=model.alternative_generic,
            alternative_specific=model.alternative_specific,
            initial_result=initial_result,
            smooth_col_ranges=smooth_col_ranges,
            smooth_penalties=smooth_penalties,
            smooth_lambdas=trial_lambdas,
            bound_nonneg_indices=model._bound_nonneg_indices,
            bound_nonpos_indices=model._bound_nonpos_indices,
        )
        result_total_edf = result.total_edf
        return result, _weighted_multinomial_gcv(
            float(result.deviance),
            weight_sum,
            None if result_total_edf is None else float(result_total_edf),
        )

    previous_selected = None
    candidate_gcvs_by_term: dict[int, list[tuple[float, float]]] = {}
    for outer_iter in range(int(max_lambda_iter)):
        start_lambdas = lambdas.copy()
        for term_idx in range(len(smooth_terms)):
            term_best_result = None
            term_best_gcv = float("inf")
            term_best_lambda = lambdas[term_idx]
            previous_candidate = previous_selected
            term_candidate_gcvs: list[tuple[float, float]] = []
            for candidate in candidate_lambdas:
                trial = lambdas.copy()
                trial[term_idx] = float(candidate)
                try:
                    candidate_result, candidate_gcv = fit_candidate(trial, previous_candidate)
                    candidate_fit_count += 1
                except (ValueError, ValidationError, RuntimeError) as exc:
                    if verbose:
                        print(
                            f"  smooth lambda candidate term={term_idx} "
                            f"lambda={candidate:.6g} failed: {exc!r}"
                        )
                    previous_candidate = None
                    continue
                previous_candidate = candidate_result
                term_candidate_gcvs.append((float(candidate), float(candidate_gcv)))
                if candidate_gcv < term_best_gcv:
                    term_best_gcv = candidate_gcv
                    term_best_lambda = float(candidate)
                    term_best_result = candidate_result
            if term_best_result is None:
                raise ValidationError(
                    "multinomial smooth tuning produced no finite GCV candidates; "
                    "check the smooth terms, lambda range, and convergence settings."
                )
            lambdas[term_idx] = term_best_lambda
            candidate_gcvs_by_term[term_idx] = term_candidate_gcvs
            previous_selected = term_best_result
            if term_best_gcv < best_gcv:
                best_gcv = term_best_gcv
                best_result = term_best_result

        max_rel_change = float(
            np.max(np.abs(lambdas - start_lambdas) / np.maximum(1.0, np.abs(start_lambdas)))
        )
        if verbose:
            print(
                f"Multinomial smooth outer={outer_iter + 1} "
                f"lambdas={lambdas.tolist()} gcv={best_gcv:.6g}"
            )
        if max_rel_change < 1e-4:
            converged = True
            break

    final_result = _fit_multinomial_arrays(
        y_codes=model.y_codes,
        x=model.X,
        n_classes=len(model.classes_),
        reference_index=model.reference_index_,
        availability=model.availability,
        offset=model.offset,
        weights=model.weights,
        alpha=0.0,
        l1_ratio=0.0,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=model.intercept,
        standardize=False,
        compute_covariance=compute_covariance,
        store_design_matrix=store_design_matrix,
        verbose=verbose,
        hessian_memory_limit_bytes=hessian_memory_limit_bytes,
        max_dense_parameters=max_dense_parameters,
        alternative_generic=model.alternative_generic,
        alternative_specific=model.alternative_specific,
        initial_result=best_result,
        smooth_col_ranges=smooth_col_ranges,
        smooth_penalties=smooth_penalties,
        smooth_lambdas=lambdas,
        bound_nonneg_indices=model._bound_nonneg_indices,
        bound_nonpos_indices=model._bound_nonpos_indices,
    )
    raw_total_edf = getattr(final_result, "total_edf", None)
    if raw_total_edf is None or not np.isfinite(raw_total_edf):
        raise ValidationError("multinomial smooth fit did not return a finite total EDF.")
    total_edf = float(raw_total_edf)
    gcv = _weighted_multinomial_gcv(float(final_result.deviance), weight_sum, total_edf)
    smooth_edfs = np.asarray(final_result.smooth_edfs, dtype=np.float64)
    if smooth_edfs.shape != (len(smooth_terms),):
        raise ValidationError(
            "multinomial smooth fit returned inconsistent EDF metadata: "
            f"expected {len(smooth_terms)} values, got {smooth_edfs.shape}."
        )
    smooth_results = []
    for idx, term in enumerate(smooth_terms):
        start, end = smooth_col_ranges[idx]
        term._lambda = float(lambdas[idx])
        term._edf = float(smooth_edfs[idx])
        smooth_results.append(
            MultinomialSmoothTermResult(
                variable=term.var_name,
                spline_type=term.spline_type,
                k=int(term.df if term.df is not None else end - start),
                lambda_=float(lambdas[idx]),
                edf=float(smooth_edfs[idx]),
                gcv=gcv,
                col_start=start,
                col_end=end,
            )
        )

    profile = {
        "method": "gcv_coordinate_search",
        "n_lambda": int(n_lambda),
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
        "candidate_fit_count": int(candidate_fit_count),
        "max_lambda_iter": int(max_lambda_iter),
        "converged": bool(converged),
        "warm_start": True,
        "selected_lambdas": [float(value) for value in lambdas],
        "gcv": float(gcv),
        "total_edf": float(total_edf),
        "candidate_gcvs": {
            smooth_terms[idx].var_name: [[float(lam), float(g)] for lam, g in candidates]
            for idx, candidates in candidate_gcvs_by_term.items()
        },
    }
    return final_result, smooth_results, total_edf, gcv, profile


def _normalize_multinomial_cv_alphas(
    model: MultinomialDict,
    fold_designs: list[MultinomialFoldDesign],
    effective_l1_ratio: float,
    *,
    alphas: np.ndarray | list[float] | None,
    n_alphas: int,
    alpha_min_ratio: float,
    include_unregularized: bool,
    standardize: bool,
) -> np.ndarray:
    if alphas is not None:
        arr = np.asarray(alphas, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValidationError("alphas must be a non-empty one-dimensional sequence.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValidationError("alphas must be finite and non-negative.")
        values = sorted({float(value) for value in arr}, reverse=True)
        return np.asarray(values, dtype=np.float64)

    if n_alphas <= 0:
        raise ValidationError("n_alphas must be positive.")
    if alpha_min_ratio <= 0.0 or alpha_min_ratio >= 1.0 or not np.isfinite(alpha_min_ratio):
        raise ValidationError("alpha_min_ratio must be finite and between 0 and 1.")

    from rustystats.regularization_path import generate_alpha_path

    fold_alpha_maxes: list[float] = []
    for fold_idx, fold in enumerate(fold_designs):
        try:
            alpha_max = _multinomial_alpha_max_from_arrays(
                fold.y_train,
                fold.x_train,
                fold.alternative_generic_train,
                fold.alternative_specific_train,
                fold.availability_train,
                fold.offset_train,
                fold.weights_train,
                len(model.classes_),
                model.reference_index_,
                effective_l1_ratio,
                fit_intercept=model.intercept,
                standardize=standardize,
            )
        except (ValidationError, ValueError, RuntimeError) as exc:
            if model._target_encoding_state is not None:
                alpha_max = ALPHA_MAX_FLOOR
            else:
                raise ValidationError(
                    f"failed to compute alpha_max for CV fold {fold_idx}: {exc}"
                ) from exc
        fold_alpha_maxes.append(alpha_max)
    alpha_max = max(fold_alpha_maxes) if fold_alpha_maxes else ALPHA_MAX_FLOOR
    path = list(generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio))
    if include_unregularized and 0.0 not in path:
        path.append(0.0)
    return np.asarray(path, dtype=np.float64)


def _fit_multinomial_cv_path(
    model: MultinomialDict,
    *,
    regularization: str,
    l1_ratio: float,
    cv: int,
    selection: str,
    n_alphas: int,
    alphas: np.ndarray | list[float] | None,
    alpha_min_ratio: float,
    cv_seed: int | None,
    include_unregularized: bool,
    max_iter: int,
    tol: float,
    standardize: bool,
    verbose: bool,
    hessian_memory_limit_bytes: int,
    max_dense_parameters: int,
) -> Any:
    from rustystats.regularization_path import (
        RegularizationPathInfo,
        RegularizationPathResult,
        select_optimal_alpha,
    )

    if regularization is None:
        raise ValidationError(
            "When cv is specified, regularization must be 'ridge', 'lasso', or 'elastic_net'."
        )
    _unused_alpha, effective_l1_ratio = _resolve_regularization(1.0, l1_ratio, regularization)
    cv_seed = DEFAULT_CV_SEED if cv_seed is None else int(cv_seed)
    folds = _stratified_multinomial_cv_folds(
        model.y_codes,
        model.weights,
        len(model.classes_),
        int(cv),
        cv_seed,
    )
    # Build each fold's design (including the expensive ordered/permutation TE
    # encode) exactly once and reuse it for both alpha-max computation and the
    # per-fold fit loop, rather than rebuilding it on the default-grid path.
    fold_designs = [
        build_multinomial_fold_design(model, train_idx, val_idx, seed=cv_seed)
        for train_idx, val_idx in folds
    ]
    for fold_idx, fold in enumerate(fold_designs):
        _validate_multinomial_dense_fit_size(
            n_shared=fold.x_train.shape[1],
            n_classes=len(model.classes_),
            n_alt_generic=fold.alternative_generic_train.shape[2],
            n_alt_specific=fold.alternative_specific_train.shape[2],
            hessian_memory_limit_bytes=hessian_memory_limit_bytes,
            max_dense_parameters=max_dense_parameters,
            context=f"CV fold {fold_idx}",
        )
    candidate_alphas = _normalize_multinomial_cv_alphas(
        model,
        fold_designs,
        effective_l1_ratio,
        alphas=alphas,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        include_unregularized=include_unregularized,
        standardize=standardize,
    )
    if verbose:
        print(
            f"Multinomial CV: {regularization}, {len(folds)} folds, {len(candidate_alphas)} alphas"
        )

    fold_scores: dict[float, list[float]] = {float(alpha): [] for alpha in candidate_alphas}
    nonzero_counts: dict[float, list[int]] = {float(alpha): [] for alpha in candidate_alphas}
    max_coefs: dict[float, list[float]] = {float(alpha): [] for alpha in candidate_alphas}

    for fold_idx, fold in enumerate(fold_designs):
        fold_nonneg_indices, fold_nonpos_indices, _shared_nonneg, _shared_nonpos = (
            _multinomial_bound_indices(
                fold.feature_names,
                n_classes=len(model.classes_),
                reference_index=model.reference_index_,
            )
        )
        previous_fold_result = None
        for alpha in candidate_alphas:
            alpha_value = float(alpha)
            fit_l1_ratio = 0.0 if alpha_value == 0.0 else effective_l1_ratio
            try:
                fold_result = _fit_multinomial_arrays(
                    y_codes=fold.y_train,
                    x=fold.x_train,
                    n_classes=len(model.classes_),
                    reference_index=model.reference_index_,
                    availability=fold.availability_train,
                    offset=fold.offset_train,
                    weights=fold.weights_train,
                    alpha=alpha_value,
                    l1_ratio=fit_l1_ratio,
                    max_iter=max_iter,
                    tol=tol,
                    fit_intercept=model.intercept,
                    standardize=standardize,
                    compute_covariance=False,
                    store_design_matrix=False,
                    verbose=False,
                    hessian_memory_limit_bytes=hessian_memory_limit_bytes,
                    max_dense_parameters=max_dense_parameters,
                    alternative_generic=fold.alternative_generic_train,
                    alternative_specific=fold.alternative_specific_train,
                    initial_result=previous_fold_result,
                    bound_nonneg_indices=fold_nonneg_indices,
                    bound_nonpos_indices=fold_nonpos_indices,
                )
                previous_fold_result = fold_result
                probabilities = _multinomial_probabilities_from_result(
                    fold_result,
                    fold.x_val,
                    fold.alternative_generic_val,
                    fold.alternative_specific_val,
                    fold.availability_val,
                    fold.offset_val,
                    len(model.classes_),
                    model.reference_index_,
                )
                score = _multinomial_validation_deviance(
                    probabilities,
                    fold.y_val,
                    fold.weights_val,
                )
            except (ValueError, ValidationError, RuntimeError) as exc:
                if verbose:
                    print(f"  fold {fold_idx}, alpha={alpha_value:.6g} failed: {exc!r}")
                score = float("inf")
                fold_result = None
            fold_scores[alpha_value].append(score)
            if fold_result is not None and np.isfinite(score):
                flat = np.concatenate(
                    [
                        np.asarray(fold_result.params, dtype=np.float64).ravel(),
                        np.asarray(
                            fold_result.alternative_generic_coefficients, dtype=np.float64
                        ).ravel(),
                        np.asarray(
                            fold_result.alternative_specific_coefficients, dtype=np.float64
                        ).ravel(),
                    ]
                )
                nonzero_counts[alpha_value].append(int(np.sum(np.abs(flat) > 1e-10)))
                max_coefs[alpha_value].append(float(np.max(np.abs(flat))) if flat.size else 0.0)

    path_results = []
    for alpha in candidate_alphas:
        alpha_value = float(alpha)
        scores = fold_scores[alpha_value]
        finite_scores = [score for score in scores if np.isfinite(score)]
        if len(finite_scores) != len(folds):
            continue
        path_results.append(
            RegularizationPathResult(
                alpha=alpha_value,
                l1_ratio=0.0 if alpha_value == 0.0 else effective_l1_ratio,
                cv_deviance_mean=float(np.mean(finite_scores)),
                cv_deviance_se=float(
                    np.std(finite_scores, ddof=1) / math.sqrt(len(finite_scores))
                    if len(finite_scores) > 1
                    else 0.0
                ),
                n_nonzero=round(np.mean(nonzero_counts[alpha_value]))
                if nonzero_counts[alpha_value]
                else 0,
                max_coef=float(np.max(max_coefs[alpha_value])) if max_coefs[alpha_value] else 0.0,
            )
        )
    if not path_results:
        raise ValidationError(
            "multinomial CV produced no finite validation deviances; check the data, "
            "fold count, regularization grid, and convergence settings."
        )
    best = select_optimal_alpha(path_results, selection)
    if best.alpha == 0.0:
        regularization_type = "none"
    elif effective_l1_ratio >= 1.0:
        regularization_type = "lasso"
    elif effective_l1_ratio <= 0.0:
        regularization_type = "ridge"
    else:
        regularization_type = "elastic_net"
    return RegularizationPathInfo(
        selected_alpha=best.alpha,
        selected_l1_ratio=best.l1_ratio,
        cv_deviance=best.cv_deviance_mean,
        cv_deviance_se=best.cv_deviance_se,
        selection_method=selection,
        regularization_type=regularization_type,
        path=path_results,
        n_folds=len(folds),
        cv_max_iter=max_iter,
        cv_tol=tol,
        fold_safe_target_encoding=model._target_encoding_state is not None,
        cv_fold_scores={alpha: list(map(float, scores)) for alpha, scores in fold_scores.items()},
        cv_scoring_objective="weighted_mean_multinomial_deviance",
        cv_profile={
            "n_alphas": len(candidate_alphas),
            "n_folds": len(folds),
            "cv_seed": cv_seed,
            "candidate_fit_count": len(candidate_alphas) * len(folds),
            "within_fold_warm_start": True,
        },
    )


def _full_data_multinomial_warm_start_result(
    model: MultinomialDict,
    path_info: Any | None,
    *,
    max_iter: int,
    tol: float,
    standardize: bool,
    verbose: bool,
    hessian_memory_limit_bytes: int,
    max_dense_parameters: int,
) -> Any | None:
    if path_info is None or not path_info.path:
        return None

    selected_alpha = float(path_info.selected_alpha)
    candidate_alphas = sorted(
        {float(row.alpha) for row in path_info.path if float(row.alpha) >= selected_alpha},
        reverse=True,
    )
    if selected_alpha not in candidate_alphas:
        candidate_alphas.append(selected_alpha)
        candidate_alphas.sort(reverse=True)

    previous_result = None
    for alpha_value in candidate_alphas:
        fit_l1_ratio = 0.0 if alpha_value == 0.0 else float(path_info.selected_l1_ratio)
        try:
            previous_result = _fit_multinomial_arrays(
                y_codes=model.y_codes,
                x=model.X,
                n_classes=len(model.classes_),
                reference_index=model.reference_index_,
                availability=model.availability,
                offset=model.offset,
                weights=model.weights,
                alpha=alpha_value,
                l1_ratio=fit_l1_ratio,
                max_iter=max_iter,
                tol=tol,
                fit_intercept=model.intercept,
                standardize=standardize,
                compute_covariance=False,
                store_design_matrix=False,
                verbose=False,
                hessian_memory_limit_bytes=hessian_memory_limit_bytes,
                max_dense_parameters=max_dense_parameters,
                alternative_generic=model.alternative_generic,
                alternative_specific=model.alternative_specific,
                initial_result=previous_result,
                bound_nonneg_indices=model._bound_nonneg_indices,
                bound_nonpos_indices=model._bound_nonpos_indices,
            )
        except (ValueError, ValidationError, RuntimeError) as exc:
            if verbose:
                print(f"  full-data warm start alpha={alpha_value:.6g} failed: {exc!r}")
            break
        if alpha_value == selected_alpha:
            break
    return previous_result


def _target_encoding_options(
    spec: dict[str, Any],
    *,
    context: str,
    allowed_keys: set[str] | None = None,
) -> tuple[float, int]:
    if "mode" in spec:
        raise ValidationError(
            "multinomial target_encoding mode is not configurable yet; only the default "
            "alternative_specific_diagonal encoding is implemented, so omit 'mode'."
        )
    if allowed_keys is not None:
        unknown = sorted(set(spec) - allowed_keys)
        if unknown:
            raise ValidationError(
                f"Unknown key(s) in target_encoding spec ({context}): {unknown}. "
                f"Valid keys are: {sorted(allowed_keys)}."
            )
    try:
        prior_weight = float(spec.get("prior_weight", DEFAULT_PRIOR_WEIGHT))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"target_encoding prior_weight must be numeric ({context}).") from exc
    try:
        n_permutations = int(spec.get("n_permutations", DEFAULT_N_PERMUTATIONS))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"target_encoding n_permutations must be an integer ({context})."
        ) from exc
    if not np.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValidationError(
            f"target_encoding prior_weight must be finite and non-negative ({context})."
        )
    if n_permutations <= 0:
        raise ValidationError(f"target_encoding n_permutations must be positive ({context}).")
    return prior_weight, n_permutations


def _validate_supported_term_spec(var_name: str, spec: dict[str, Any], *, context: str) -> None:
    term_type = spec.get("type", "linear")
    monotonicity = spec.get("monotonicity")
    if term_type in {"ms", "s"}:
        raise ValidationError(
            f"{term_type} smooth/monotonic spline terms are not yet supported for "
            "multinomial_dict. Use bs/ns smooth terms without monotonicity."
        )
    smooth_requested = term_type in {"bs", "ns"} and (
        spec.get("k") is not None or (spec.get("df") is None and spec.get("knots") is None)
    )
    if monotonicity is not None:
        if monotonicity not in {"increasing", "decreasing"}:
            raise ValidationError(
                f"monotonicity must be 'increasing' or 'decreasing' for multinomial_dict "
                f"({context} term {var_name!r})."
            )
        if context != "main":
            raise ValidationError(
                f"monotonicity constraints are only supported for multinomial main effects "
                f"({context} term {var_name!r})."
            )
        if term_type in {"linear", "expression"}:
            pass
        elif term_type == "bs":
            if smooth_requested:
                raise ValidationError(
                    "monotonic smooth bs terms are not yet supported for multinomial_dict. "
                    f"Use a fixed basis with df= or knots= ({context} term {var_name!r})."
                )
        elif term_type == "ns":
            raise ValidationError(
                "monotonicity constraints are not supported for multinomial natural splines "
                f"({context} term {var_name!r}); use type='bs' with df= or knots= instead."
            )
        elif term_type in {"categorical", "target_encoding", "frequency_encoding"}:
            raise ValidationError(
                f"monotonicity constraints are not supported for multinomial {term_type} terms "
                f"({context} term {var_name!r})."
            )
        else:
            raise ValidationError(
                f"term type {term_type!r} is not supported for multinomial monotonicity "
                f"({context} term {var_name!r})."
            )
    if term_type in {"bs", "ns"}:
        if smooth_requested and context != "main":
            raise ValidationError(
                f"automatic smooth penalties for {term_type} terms are only supported "
                f"as main effects in multinomial_dict ({context} term {var_name!r}). "
                "Use df= or knots= for a fixed interaction basis."
            )
    if term_type == "target_encoding":
        _target_encoding_options(
            spec,
            context=f"{context} term {var_name!r}",
            allowed_keys={"type", "prior_weight", "n_permutations", "variable"},
        )
        return
    if term_type not in {"linear", "categorical", "bs", "ns", "frequency_encoding", "expression"}:
        raise ValidationError(f"term type {term_type!r} is not supported for multinomial_dict.")


def _validate_supported_terms(
    terms: dict[str, dict[str, Any]],
    interactions: list[dict[str, Any]] | None,
) -> None:
    for var_name, spec in terms.items():
        _validate_supported_term_spec(var_name, spec, context="main")

    reserved = {
        "include_main",
        "target_encoding",
        "frequency_encoding",
        "prior_weight",
        "n_permutations",
        "mode",
    }
    for interaction in interactions or []:
        is_target_encoded_interaction = bool(interaction.get("target_encoding"))
        if is_target_encoded_interaction:
            _target_encoding_options(interaction, context="interaction")
        for var_name, spec in interaction.items():
            if var_name in reserved:
                continue
            if not isinstance(spec, dict):
                raise ValidationError(f"interaction spec for {var_name!r} must be a dict.")
            if not is_target_encoded_interaction and spec.get("type") == "target_encoding":
                raise ValidationError(
                    "target_encoding factors inside ordinary interactions are not yet supported "
                    "for multinomial_dict. Use an interaction with target_encoding=True instead."
                )
            _validate_supported_term_spec(var_name, spec, context="interaction")


def _resolve_classes(
    response_values: list[str],
    response_series: Any,
    classes: list[str] | None,
    reference: str | None,
) -> tuple[list[str], str, dict[str, int], np.ndarray]:
    if classes is None:
        resolved_classes = _derive_classes(response_series)
    else:
        resolved_classes = [str(value) for value in classes]

    if len(resolved_classes) < 2:
        raise ValidationError("multinomial_dict requires at least two classes.")
    if len(set(resolved_classes)) != len(resolved_classes):
        raise ValidationError("classes must be unique after string conversion.")

    class_to_code = {label: idx for idx, label in enumerate(resolved_classes)}
    missing = sorted(set(response_values) - set(resolved_classes))
    if missing:
        raise ValidationError(f"observed response labels are not present in classes: {missing}.")
    empty = [label for label in resolved_classes if label not in set(response_values)]
    if empty:
        raise ValidationError(
            "classes with no observed rows are not supported in Phase 1: "
            f"{empty}. Remove them or wait for prediction-only class metadata support."
        )

    resolved_reference = str(reference) if reference is not None else resolved_classes[0]
    if resolved_reference not in class_to_code:
        raise ValidationError(
            f"reference={resolved_reference!r} is not in classes {resolved_classes!r}."
        )

    y_codes = np.asarray([class_to_code[value] for value in response_values], dtype=np.int64)
    return resolved_classes, resolved_reference, class_to_code, y_codes


def _resolve_weights(
    data: Any,
    weights: str | np.ndarray | None,
    class_weights: dict[str, float] | None,
    response_values: list[str],
) -> tuple[np.ndarray | None, bool, np.ndarray | None]:
    """Resolve fit weights.

    Returns ``(effective_weights, is_class_weighted, row_weights)`` where
    ``effective_weights`` folds the class multipliers (what the solver fits on)
    and ``row_weights`` is the unfolded observation-weight vector (``None`` when
    no row weights were supplied). Diagnostics use ``row_weights`` so the
    reported class mix reflects the data distribution rather than the
    class-reweighted training objective.
    """
    row_weights = None
    if weights is not None:
        if isinstance(weights, str):
            row_weights = data[weights].to_numpy().astype(np.float64)
        else:
            row_weights = _as_float_array(weights, name="weights", length=len(data))
        if np.any(~np.isfinite(row_weights)) or np.any(row_weights < 0.0):
            raise ValidationError("weights must be finite and non-negative.")

    if class_weights is None:
        return row_weights, False, row_weights

    resolved = {str(key): float(value) for key, value in class_weights.items()}
    bad = [key for key, value in resolved.items() if not np.isfinite(value) or value < 0.0]
    if bad:
        raise ValidationError(f"class_weights must be finite and non-negative; bad keys: {bad}.")

    class_multiplier = np.asarray([resolved.get(label, 1.0) for label in response_values])
    effective = class_multiplier if row_weights is None else row_weights * class_multiplier
    return effective.astype(np.float64, copy=False), True, row_weights


def _resolve_class_matrix(
    data: Any,
    spec: dict[str, Any] | np.ndarray | None,
    classes: list[str],
    *,
    default: bool | float,
    dtype: type,
    name: str,
    allow_arrays: bool,
) -> np.ndarray:
    n = len(data)
    k = len(classes)

    if spec is None:
        return np.full((n, k), default, dtype=dtype)

    if isinstance(spec, np.ndarray):
        if not allow_arrays:
            raise PredictionError(
                f"{name} was fit from an array, which is fit-time data. Pass {name}= for "
                "prediction data, or fit with column names/scalars."
            )
        arr = np.asarray(spec, dtype=dtype)
        if arr.shape != (n, k):
            raise ValidationError(f"{name} array must have shape ({n}, {k}); got {arr.shape}.")
        return arr

    if not isinstance(spec, dict):
        raise ValidationError(f"{name} must be a dict, an array, or None.")

    out = np.full((n, k), default, dtype=dtype)
    for class_idx, class_label in enumerate(classes):
        value = spec.get(class_label, default)
        if isinstance(value, str):
            if value not in data.columns:
                raise PredictionError(f"{name} column {value!r} is not present in data.")
            arr = data[value].to_numpy()
            if dtype is bool:
                out[:, class_idx] = np.asarray(arr, dtype=bool)
            else:
                out[:, class_idx] = _as_float_array(arr, name=f"{name}[{class_label}]")
        elif isinstance(value, (bool, np.bool_)):
            out[:, class_idx] = bool(value)
        elif _array_like_not_string(value):
            if not allow_arrays:
                raise PredictionError(
                    f"{name}[{class_label!r}] was fit from an array, which is fit-time data. "
                    f"Pass {name}= for prediction data, or fit with a column name/scalar."
                )
            arr = np.asarray(value, dtype=dtype)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValidationError(
                    f"{name}[{class_label!r}] must have length {n}; got shape {arr.shape}."
                )
            out[:, class_idx] = arr
        else:
            out[:, class_idx] = dtype(value)

    return out


def _slice_class_matrix_override(
    spec: Any,
    *,
    start: int,
    stop: int,
    n_rows: int,
    n_classes: int,
    name: str,
) -> Any:
    if spec is None:
        return None
    if isinstance(spec, np.ndarray):
        arr = np.asarray(spec)
        if arr.shape != (n_rows, n_classes):
            raise ValidationError(
                f"{name} array must have shape ({n_rows}, {n_classes}); got {arr.shape}."
            )
        return arr[start:stop, :]
    if not isinstance(spec, dict):
        return spec

    sliced = {}
    for class_label, value in spec.items():
        if isinstance(value, (str, bool, np.bool_)):
            sliced[class_label] = value
        elif _array_like_not_string(value):
            arr = np.asarray(value)
            if arr.ndim != 1 or arr.shape[0] != n_rows:
                raise ValidationError(
                    f"{name}[{class_label!r}] must have length {n_rows}; got shape {arr.shape}."
                )
            sliced[class_label] = arr[start:stop]
        else:
            sliced[class_label] = value
    return sliced


def _masked_softmax(logits: np.ndarray, availability: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    availability = np.asarray(availability, dtype=bool)
    if logits.shape != availability.shape:
        raise PredictionError(
            f"logits shape {logits.shape} does not match availability shape {availability.shape}."
        )
    masked = np.where(availability, logits, -np.inf)
    max_eta = np.max(masked, axis=1, keepdims=True)
    if not np.all(np.isfinite(max_eta)):
        raise PredictionError("availability leaves at least one prediction row with no classes.")
    exp_eta = np.where(availability, np.exp(masked - max_eta), 0.0)
    denom = exp_eta.sum(axis=1, keepdims=True)
    return exp_eta / denom


def _weighted_probability_mix(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValidationError("weights must have positive total.")
    return (probabilities * weights[:, None]).sum(axis=0) / total_weight


def _share_based_null_deviance(class_counts: dict[str, float], total_weight: float) -> float | None:
    """Intercept-only null deviance from weighted class counts: -2 sum_k n_k log(n_k/N)."""
    if total_weight <= 0.0:
        return None
    log_likelihood = 0.0
    for count in class_counts.values():
        if count > 0.0:
            log_likelihood += count * math.log(count / total_weight)
    return -2.0 * log_likelihood


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _optional_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None and np.isfinite(value)]
    if not defined:
        return None
    return float(np.mean(defined))


def _stringify_factor_values(values: Any) -> np.ndarray:
    labels: list[str] = []
    for value in values:
        if value is None:
            labels.append("<null>")
        else:
            try:
                if bool(np.asarray(value).shape == ()) and bool(np.isnan(value)):  # type: ignore[arg-type]
                    labels.append("<null>")
                    continue
            except Exception:
                pass
            labels.append(str(value))
    return np.asarray(labels, dtype=object)


def _continuous_factor_bins(values: Any, *, max_bins: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    labels = np.full(arr.shape[0], "<null>", dtype=object)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return labels
    finite_values = arr[finite]
    unique = np.unique(finite_values)
    if unique.size <= max_bins:
        labels[finite] = [f"{value:.6g}" for value in finite_values]
        return labels

    edges = np.unique(np.quantile(finite_values, np.linspace(0.0, 1.0, max_bins + 1)))
    if edges.size <= 2:
        labels[finite] = [f"{value:.6g}" for value in finite_values]
        return labels
    bin_codes = np.searchsorted(edges[1:-1], finite_values, side="right")
    bin_labels = []
    for code in bin_codes:
        left = edges[code]
        right = edges[code + 1]
        bin_labels.append(f"[{left:.6g}, {right:.6g}]")
    labels[finite] = bin_labels
    return labels


@dataclass
class _DeserializedMultinomialResult:
    params: np.ndarray
    alternative_generic_coefficients: np.ndarray
    alternative_specific_coefficients: np.ndarray
    fitted_probabilities: np.ndarray
    linear_predictor: np.ndarray
    log_likelihood: float
    deviance: float
    null_deviance: float
    iterations: int
    converged: bool
    covariance_unscaled: np.ndarray | None
    prior_weights: np.ndarray
    y_codes: np.ndarray
    reference_index: int
    warnings: list[str]
    solver_status: str
    alpha: float
    l1_ratio: float
    fit_intercept: bool
    smooth_edfs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    total_edf: float | None = None

    @property
    def coefficients(self) -> np.ndarray:
        return self.params

    @property
    def coef_matrix(self) -> np.ndarray:
        return self.params

    @property
    def fittedvalues(self) -> np.ndarray:
        return self.fitted_probabilities

    @property
    def cov_params_unscaled(self) -> np.ndarray | None:
        return self.covariance_unscaled

    @property
    def nobs(self) -> int:
        return int(self.y_codes.shape[0])


@dataclass
class MultinomialDatasetDiagnostics:
    """Model diagnostics for one evaluated multinomial dataset."""

    name: str
    classes: list[str]
    nobs: int
    total_weight: float
    log_loss: float
    deviance: float
    accuracy: float
    top_2_accuracy: float
    balanced_accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    confusion_matrix: np.ndarray
    actual_class_counts: dict[str, float]
    predicted_class_counts: dict[str, float]
    actual_class_mix: dict[str, float]
    predicted_class_mix: dict[str, float]
    class_mix_error: dict[str, float]
    per_class_metrics: dict[str, dict[str, float | None]]
    class_calibration: dict[str, list[dict[str, Any]]]
    expected_calibration_error_by_class: dict[str, float]
    multiclass_expected_calibration_error: float
    reliability_by_winning_class: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": list(self.classes),
            "nobs": self.nobs,
            "total_weight": self.total_weight,
            "log_loss": self.log_loss,
            "deviance": self.deviance,
            "accuracy": self.accuracy,
            "top_2_accuracy": self.top_2_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "actual_class_counts": dict(self.actual_class_counts),
            "predicted_class_counts": dict(self.predicted_class_counts),
            "actual_class_mix": dict(self.actual_class_mix),
            "predicted_class_mix": dict(self.predicted_class_mix),
            "class_mix_error": dict(self.class_mix_error),
            "per_class_metrics": self.per_class_metrics,
            "class_calibration": self.class_calibration,
            "expected_calibration_error_by_class": dict(self.expected_calibration_error_by_class),
            "multiclass_expected_calibration_error": self.multiclass_expected_calibration_error,
            "reliability_by_winning_class": self.reliability_by_winning_class,
        }


@dataclass
class MultinomialDiagnostics:
    """Pricing-grade diagnostics for a fitted multinomial model."""

    model_type: str
    classes: list[str]
    reference: str
    nobs: int
    n_params: int
    log_likelihood: float
    deviance: float
    null_deviance: float
    mcfadden_pseudo_r2: float | None
    aic: float | None
    bic: float | None
    log_loss: float
    accuracy: float
    top_2_accuracy: float
    balanced_accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    confusion_matrix: np.ndarray
    actual_class_counts: dict[str, float]
    predicted_class_counts: dict[str, float]
    actual_class_mix: dict[str, float]
    predicted_class_mix: dict[str, float]
    class_mix_error: dict[str, float]
    per_class_metrics: dict[str, dict[str, float | None]]
    class_calibration: dict[str, list[dict[str, Any]]]
    expected_calibration_error_by_class: dict[str, float]
    multiclass_expected_calibration_error: float
    reliability_by_winning_class: dict[str, dict[str, Any]]
    factor_diagnostics: list[dict[str, Any]]
    train: MultinomialDatasetDiagnostics
    test: MultinomialDatasetDiagnostics | None
    train_test_comparison: dict[str, float | None] | None
    inference_status: str
    solver_status: str
    converged: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "classes": list(self.classes),
            "reference": self.reference,
            "nobs": self.nobs,
            "n_params": self.n_params,
            "log_likelihood": self.log_likelihood,
            "deviance": self.deviance,
            "null_deviance": self.null_deviance,
            "mcfadden_pseudo_r2": self.mcfadden_pseudo_r2,
            "aic": self.aic,
            "bic": self.bic,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "top_2_accuracy": self.top_2_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "actual_class_counts": dict(self.actual_class_counts),
            "predicted_class_counts": dict(self.predicted_class_counts),
            "actual_class_mix": dict(self.actual_class_mix),
            "predicted_class_mix": dict(self.predicted_class_mix),
            "class_mix_error": dict(self.class_mix_error),
            "per_class_metrics": self.per_class_metrics,
            "class_calibration": self.class_calibration,
            "expected_calibration_error_by_class": dict(self.expected_calibration_error_by_class),
            "multiclass_expected_calibration_error": self.multiclass_expected_calibration_error,
            "reliability_by_winning_class": self.reliability_by_winning_class,
            "factor_diagnostics": list(self.factor_diagnostics),
            "train": self.train.to_dict(),
            "test": None if self.test is None else self.test.to_dict(),
            "train_test_comparison": self.train_test_comparison,
            "inference_status": self.inference_status,
            "solver_status": self.solver_status,
            "converged": self.converged,
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class MultinomialInterceptCalibration:
    """Vector-intercept calibration for a fitted multinomial model.

    The calibration is an additive class-level logit shift. The reference class
    shift is fixed at zero for identifiability; fitted model coefficients are
    never mutated.
    """

    classes: list[str]
    reference: str
    shifts: dict[str, float]
    actual_class_mix: dict[str, float] = field(default_factory=dict)
    base_predicted_class_mix: dict[str, float] = field(default_factory=dict)
    calibrated_class_mix: dict[str, float] = field(default_factory=dict)
    nobs: int = 0
    total_weight: float = 0.0
    iterations: int = 0
    converged: bool = True
    method: str = "vector_intercept"
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.method = str(self.method)
        if self.method != "vector_intercept":
            raise ValidationError(
                "MultinomialInterceptCalibration only supports method='vector_intercept'."
            )
        self.classes = [str(label) for label in self.classes]
        if len(self.classes) < 2:
            raise ValidationError("calibration classes must contain at least two labels.")
        if len(set(self.classes)) != len(self.classes):
            raise ValidationError("calibration classes must be unique.")
        self.reference = str(self.reference)
        if self.reference not in self.classes:
            raise ValidationError("calibration reference must be present in classes.")
        self.shifts = self._normalize_required_class_dict(self.shifts, "shifts")
        if abs(self.shifts[self.reference]) > 1e-12:
            raise ValidationError("calibration reference shift must be 0.0.")
        self.actual_class_mix = self._normalize_optional_class_dict(
            self.actual_class_mix, "actual_class_mix"
        )
        self.base_predicted_class_mix = self._normalize_optional_class_dict(
            self.base_predicted_class_mix, "base_predicted_class_mix"
        )
        self.calibrated_class_mix = self._normalize_optional_class_dict(
            self.calibrated_class_mix, "calibrated_class_mix"
        )
        self.nobs = int(self.nobs)
        if self.nobs < 0:
            raise ValidationError("calibration nobs must be non-negative.")
        self.total_weight = float(self.total_weight)
        if not np.isfinite(self.total_weight) or self.total_weight < 0.0:
            raise ValidationError("calibration total_weight must be finite and non-negative.")
        self.iterations = int(self.iterations)
        if self.iterations < 0:
            raise ValidationError("calibration iterations must be non-negative.")
        self.converged = bool(self.converged)
        self.warnings = [str(message) for message in self.warnings]

    def _normalize_required_class_dict(
        self, values: dict[str, float], name: str
    ) -> dict[str, float]:
        if not isinstance(values, dict):
            raise ValidationError(f"calibration {name} must be a dict.")
        normalized = {str(key): float(value) for key, value in values.items()}
        missing = [label for label in self.classes if label not in normalized]
        extras = sorted(set(normalized) - set(self.classes))
        if missing or extras:
            raise ValidationError(
                f"calibration {name} must contain exactly the model classes; "
                f"missing={missing}, extras={extras}."
            )
        bad = [label for label, value in normalized.items() if not np.isfinite(value)]
        if bad:
            raise ValidationError(f"calibration {name} contains non-finite values: {bad}.")
        return {label: normalized[label] for label in self.classes}

    def _normalize_optional_class_dict(
        self, values: dict[str, float], name: str
    ) -> dict[str, float]:
        if values is None:
            return {}
        if not isinstance(values, dict):
            raise ValidationError(f"calibration {name} must be a dict.")
        normalized = {str(key): float(value) for key, value in values.items()}
        extras = sorted(set(normalized) - set(self.classes))
        if extras:
            raise ValidationError(f"calibration {name} contains unknown classes: {extras}.")
        bad = [label for label, value in normalized.items() if not np.isfinite(value)]
        if bad:
            raise ValidationError(f"calibration {name} contains non-finite values: {bad}.")
        return {label: normalized[label] for label in self.classes if label in normalized}

    @property
    def shift_vector(self) -> np.ndarray:
        return np.asarray([self.shifts[label] for label in self.classes], dtype=np.float64)

    def adjust_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != len(self.classes):
            raise ValidationError(
                f"logits must have shape (n_rows, {len(self.classes)}); got {logits.shape}."
            )
        return logits + self.shift_vector[None, :]

    def predict_proba_from_logits(
        self, logits: np.ndarray, availability: np.ndarray | None = None
    ) -> np.ndarray:
        adjusted = self.adjust_logits(logits)
        if availability is None:
            availability = np.ones_like(adjusted, dtype=bool)
        return _masked_softmax(adjusted, availability)

    def predict_proba(
        self,
        model: Any,
        new_data: Any,
        *,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        return_format: str = "numpy",
    ) -> np.ndarray:
        return model.predict_proba(
            new_data,
            availability=availability,
            offset=offset,
            return_format=return_format,
            calibration=self,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "classes": list(self.classes),
            "reference": self.reference,
            "shifts": dict(self.shifts),
            "actual_class_mix": dict(self.actual_class_mix),
            "base_predicted_class_mix": dict(self.base_predicted_class_mix),
            "calibrated_class_mix": dict(self.calibrated_class_mix),
            "nobs": self.nobs,
            "total_weight": self.total_weight,
            "iterations": self.iterations,
            "converged": self.converged,
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultinomialInterceptCalibration:
        if not isinstance(payload, dict):
            raise ValidationError("calibration payload must be a dict.")
        return cls(
            classes=payload["classes"],
            reference=payload["reference"],
            shifts=payload["shifts"],
            actual_class_mix=payload.get("actual_class_mix", {}),
            base_predicted_class_mix=payload.get("base_predicted_class_mix", {}),
            calibrated_class_mix=payload.get("calibrated_class_mix", {}),
            nobs=payload.get("nobs", 0),
            total_weight=payload.get("total_weight", 0.0),
            iterations=payload.get("iterations", 0),
            converged=payload.get("converged", True),
            method=payload.get("method", "vector_intercept"),
            warnings=payload.get("warnings", []),
        )


@dataclass
class MultinomialScenario:
    """Aggregate result for an alternative-specific pricing scenario."""

    classes: list[str]
    nobs: int
    total_weight: float
    base_class_mix: dict[str, float]
    scenario_class_mix: dict[str, float]
    class_mix_delta: dict[str, float]
    base_probabilities: np.ndarray
    scenario_probabilities: np.ndarray
    expected_value: dict[str, float] | None
    segment_mix: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "nobs": self.nobs,
            "total_weight": self.total_weight,
            "base_class_mix": dict(self.base_class_mix),
            "scenario_class_mix": dict(self.scenario_class_mix),
            "class_mix_delta": dict(self.class_mix_delta),
            "expected_value": None if self.expected_value is None else dict(self.expected_value),
            "segment_mix": list(self.segment_mix),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class _DeserializedBuilder(InteractionBuilder):
    def __init__(self, state: dict[str, Any]):
        self._parsed_formula = state["parsed_formula"]
        self._cat_encoding_cache = state["cat_encoding_cache"]
        self._fitted_splines = state["fitted_splines"]
        self._te_stats = state.get("te_stats", {})
        self._fe_stats = state.get("fe_stats", {})
        self.dtype = state["dtype"]
        self.data = None
        self._n = 0
        self._term_slots = state["term_slots"]


class MultinomialModel:
    """Fitted native baseline-category multinomial logit model."""

    def __init__(
        self,
        *,
        result: Any,
        response: str | None,
        classes: list[str],
        reference: str,
        feature_names: list[str],
        builder: InteractionBuilder | None,
        terms: dict[str, dict[str, Any]],
        alternative_terms: dict[str, dict[str, Any]] | None,
        interactions: list[dict[str, Any]] | None,
        input_transforms: list[dict[str, Any]] | None,
        compiled_input_transforms: list[CompiledInputTransform] | None,
        target_encoding_state: _MultinomialTargetEncodingState | None,
        availability_spec: Any,
        offset_spec: Any,
        weights_spec: str | None,
        array_availability_requires_prediction_override: bool,
        array_offset_requires_prediction_override: bool,
        inference_status: str,
        fit_row_weights: np.ndarray | None = None,
        array_weighted: bool = False,
        regularization_path_info: Any | None = None,
        smooth_results: list[MultinomialSmoothTermResult] | None = None,
        total_edf: float | None = None,
        gcv: float | None = None,
        smooth_profile: dict[str, Any] | None = None,
        constraint_metadata: list[dict[str, Any]] | None = None,
    ):
        self._result = result
        self.response = response
        self.classes_ = list(classes)
        self.reference_ = reference
        self.reference_index_ = self.classes_.index(reference)
        self.feature_names = list(feature_names)
        self._builder = builder
        self._terms_dict = copy.deepcopy(terms)
        self._alternative_terms_spec = copy.deepcopy(alternative_terms or {})
        self.alternative_generic_feature_names, self.alternative_specific_feature_names = (
            _alternative_names_by_kind(self._alternative_terms_spec)
        )
        self._target_encoding_state = copy.deepcopy(target_encoding_state)
        if self._target_encoding_state is not None:
            self.alternative_specific_feature_names.extend(
                self._target_encoding_state.feature_names
            )
        self._interactions_spec = copy.deepcopy(interactions)
        self._input_transforms = validate_input_transforms(input_transforms)
        self._compiled_input_transforms = (
            list(compiled_input_transforms)
            if compiled_input_transforms is not None
            else compile_input_transforms(self._input_transforms, assume_validated=True)
        )
        self._availability_spec = copy.deepcopy(availability_spec)
        self._offset_spec = copy.deepcopy(offset_spec)
        self._weights_spec = weights_spec
        self._array_availability_requires_prediction_override = bool(
            array_availability_requires_prediction_override
        )
        self._array_offset_requires_prediction_override = bool(
            array_offset_requires_prediction_override
        )
        self._fit_row_weights = (
            None if fit_row_weights is None else np.asarray(fit_row_weights, dtype=np.float64)
        )
        self._array_weighted = bool(array_weighted)
        self.inference_status = inference_status
        self._regularization_path_info = regularization_path_info
        self._smooth_results = list(smooth_results or [])
        self._total_edf = None if total_edf is None else float(total_edf)
        self._gcv = None if gcv is None else float(gcv)
        self._smooth_profile = copy.deepcopy(smooth_profile)
        self._constraint_metadata = copy.deepcopy(constraint_metadata or [])

    @property
    def params(self) -> np.ndarray:
        return np.asarray(self._result.params, dtype=np.float64)

    @property
    def coef_matrix(self) -> np.ndarray:
        return self.params

    @property
    def alternative_generic_coefficients(self) -> np.ndarray:
        values = getattr(self._result, "alternative_generic_coefficients", None)
        if values is None:
            return np.zeros(len(self.alternative_generic_feature_names), dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    @property
    def alternative_specific_coefficients(self) -> np.ndarray:
        values = getattr(self._result, "alternative_specific_coefficients", None)
        if values is None:
            return np.zeros(
                (len(self.classes_) - 1, len(self.alternative_specific_feature_names)),
                dtype=np.float64,
            )
        return np.asarray(values, dtype=np.float64)

    @property
    def intercepts(self) -> np.ndarray:
        intercepts = np.zeros(len(self.classes_), dtype=np.float64)
        if not self.feature_names or self.feature_names[0] != "Intercept":
            return intercepts
        block = 0
        for class_idx, class_label in enumerate(self.classes_):
            if class_label == self.reference_:
                continue
            intercepts[class_idx] = self.params[block, 0]
            block += 1
        return intercepts

    @property
    def fitted_probabilities(self) -> np.ndarray:
        return np.asarray(self._result.fitted_probabilities, dtype=np.float64)

    @property
    def linear_predictor(self) -> np.ndarray:
        return np.asarray(self._result.linear_predictor, dtype=np.float64)

    @property
    def log_likelihood(self) -> float:
        return float(self._result.log_likelihood)

    @property
    def deviance(self) -> float:
        return float(self._result.deviance)

    @property
    def null_deviance(self) -> float:
        return float(self._result.null_deviance)

    @property
    def iterations(self) -> int:
        return int(self._result.iterations)

    @property
    def converged(self) -> bool:
        return bool(self._result.converged)

    @property
    def solver_status(self) -> str:
        return str(self._result.solver_status)

    @property
    def warnings(self) -> list[str]:
        return list(self._result.warnings)

    @property
    def alpha(self) -> float:
        return float(getattr(self._result, "alpha", 0.0))

    @property
    def l1_ratio(self) -> float:
        return float(getattr(self._result, "l1_ratio", 0.0))

    @property
    def cv_deviance(self) -> float | None:
        if self._regularization_path_info is None:
            return None
        return float(self._regularization_path_info.cv_deviance)

    @property
    def cv_deviance_se(self) -> float | None:
        if self._regularization_path_info is None:
            return None
        return float(self._regularization_path_info.cv_deviance_se)

    @property
    def regularization_type(self) -> str | None:
        if self._smooth_results:
            return "smooth"
        if self._regularization_path_info is not None:
            return self._regularization_path_info.regularization_type
        if self.alpha <= 0.0:
            return "none"
        if self.l1_ratio >= 1.0:
            return "lasso"
        if self.l1_ratio > 0.0:
            return "elastic_net"
        return "ridge"

    @property
    def regularization_path(self) -> list[dict[str, Any]] | None:
        if self._regularization_path_info is None:
            return None
        return [
            {
                "alpha": float(row.alpha),
                "l1_ratio": float(row.l1_ratio),
                "cv_deviance_mean": float(row.cv_deviance_mean),
                "cv_deviance_se": float(row.cv_deviance_se),
                "n_nonzero": int(row.n_nonzero),
                "max_coef": float(row.max_coef),
            }
            for row in self._regularization_path_info.path
        ]

    @property
    def cv_selection_method(self) -> str | None:
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.selection_method

    @property
    def n_cv_folds(self) -> int | None:
        if self._regularization_path_info is None:
            return None
        return int(self._regularization_path_info.n_folds)

    @property
    def cv_convergence(self) -> dict[str, float | int] | None:
        if self._regularization_path_info is None:
            return None
        return {
            "max_iter": int(self._regularization_path_info.cv_max_iter),
            "tol": float(self._regularization_path_info.cv_tol),
        }

    @property
    def cv_fold_scores(self) -> dict[float, list[float]] | None:
        if self._regularization_path_info is None:
            return None
        scores = self._regularization_path_info.cv_fold_scores
        if scores is None:
            return None
        return {
            float(alpha): list(map(float, fold_scores)) for alpha, fold_scores in scores.items()
        }

    @property
    def cv_scoring_objective(self) -> str | None:
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_scoring_objective

    @property
    def cv_profile(self) -> dict[str, Any] | None:
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_profile

    @property
    def fold_safe_target_encoding(self) -> bool | None:
        if self._regularization_path_info is None:
            return None
        return bool(self._regularization_path_info.fold_safe_target_encoding)

    @property
    def smooth_terms(self) -> list[dict[str, Any]]:
        return [smooth.to_dict() for smooth in self._smooth_results]

    @property
    def smooth_lambdas(self) -> list[float]:
        return [float(smooth.lambda_) for smooth in self._smooth_results]

    @property
    def smooth_edfs(self) -> list[float]:
        return [float(smooth.edf) for smooth in self._smooth_results]

    @property
    def total_edf(self) -> float | None:
        return self._total_edf

    @property
    def gcv(self) -> float | None:
        return self._gcv

    @property
    def smooth_profile(self) -> dict[str, Any] | None:
        return None if self._smooth_profile is None else copy.deepcopy(self._smooth_profile)

    @property
    def constraint_metadata(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._constraint_metadata)

    @property
    def active_constraints(self) -> list[dict[str, Any]]:
        active = []
        flat_params = self.params.ravel()
        for record in self._constraint_metadata:
            parameter_indices = record.get("parameter_indices", [])
            active_indices = [
                int(idx)
                for idx in parameter_indices
                if 0 <= int(idx) < flat_params.size and abs(float(flat_params[int(idx)])) <= 1e-8
            ]
            if active_indices:
                item = copy.deepcopy(record)
                item["active_parameter_indices"] = active_indices
                active.append(item)
        return active

    @property
    def nobs(self) -> int:
        return int(self.fitted_probabilities.shape[0])

    @property
    def n_params(self) -> int:
        return int(
            self.params.size
            + self.alternative_generic_coefficients.size
            + self.alternative_specific_coefficients.size
        )

    def llf(self) -> float:
        return self.log_likelihood

    def aic(self) -> float | None:
        if (
            self.alpha > 0.0
            or "class_weighted" in self.inference_status
            or "constrained_boundary" in self.inference_status
        ):
            return None
        if self.total_edf is not None:
            return -2.0 * self.log_likelihood + 2.0 * self.total_edf
        return -2.0 * self.log_likelihood + 2.0 * self.n_params

    def bic(self) -> float | None:
        if (
            self.alpha > 0.0
            or "class_weighted" in self.inference_status
            or "constrained_boundary" in self.inference_status
        ):
            return None
        if self.total_edf is not None:
            return -2.0 * self.log_likelihood + self.total_edf * math.log(self.nobs)
        return -2.0 * self.log_likelihood + self.n_params * math.log(self.nobs)

    def _class_codes_from_values(self, values: Any, *, name: str) -> np.ndarray:
        labels = _string_labels(values)
        class_to_code = {label: idx for idx, label in enumerate(self.classes_)}
        unknown = sorted(set(labels) - set(class_to_code))
        if unknown:
            raise ValidationError(
                f"{name} contains labels not present in model classes: {unknown}."
            )
        return np.asarray([class_to_code[label] for label in labels], dtype=np.int64)

    def _diagnostic_weight_array(
        self, data: Any, weights_override: str | None = None
    ) -> np.ndarray:
        column = weights_override if weights_override is not None else self._weights_spec
        if column is None:
            if self._array_weighted:
                raise PredictionError(
                    "This model was fit with an in-memory array weight vector, which cannot be "
                    "re-resolved for supplied diagnostic data. Pass weights='<column>' to "
                    "diagnostics(), or call diagnostics() without train_data to use the fitted "
                    "weights."
                )
            return np.ones(len(data), dtype=np.float64)
        if not isinstance(column, str):
            raise ValidationError("diagnostics weights override must be a column name.")
        if column not in data.columns:
            raise PredictionError(f"weights column {column!r} is not present in data.")
        weights = _as_float_array(data[column].to_numpy(), name="weights", length=len(data))
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValidationError("weights must be finite and non-negative.")
        return weights

    def _collect_diagnostic_data(
        self,
        data: Any,
        *,
        categorical_factors: list[str],
        continuous_factors: list[str],
        weights_override: str | None = None,
    ) -> Any:
        if self.response is None:
            raise PredictionError(
                "Cannot compute diagnostics on supplied data: this model has no stored response "
                "column metadata."
            )

        requested = {self.response, *categorical_factors, *continuous_factors}
        if self._weights_spec is not None:
            requested.add(self._weights_spec)
        if isinstance(weights_override, str):
            requested.add(weights_override)
        produced = {
            spec.get("output")
            for spec in self._input_transforms
            if isinstance(spec, dict) and isinstance(spec.get("output"), str)
        }
        needed = {column for column in requested if column not in produced}
        needed |= set(input_transform_source_columns(self._input_transforms))
        collected = _collect_lazyframe(data, needed)
        if self._compiled_input_transforms:
            drop_outputs = [
                spec["output"]
                for spec in self._input_transforms
                if spec["output"] in collected.columns
            ]
            if drop_outputs:
                collected = collected.drop(drop_outputs)
            collected = apply_input_transforms(collected, self._compiled_input_transforms)
        return collected

    def _diagnostic_arrays_from_data(
        self,
        data: Any,
        *,
        categorical_factors: list[str],
        continuous_factors: list[str],
        weights_override: str | None = None,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        diagnostic_data = self._collect_diagnostic_data(
            data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            weights_override=weights_override,
        )
        probabilities = self.predict_proba(data, return_format="numpy")
        if probabilities.shape[0] != len(diagnostic_data):
            raise PredictionError("Diagnostic data and prediction row counts do not match.")
        y_codes = self._class_codes_from_values(
            diagnostic_data[self.response].to_list(), name=self.response or "response"
        )
        weights = self._diagnostic_weight_array(diagnostic_data, weights_override)
        return diagnostic_data, probabilities, y_codes, weights

    def _calibration_by_class(
        self,
        probabilities: np.ndarray,
        y_codes: np.ndarray,
        weights: np.ndarray,
        *,
        n_bins: int = 10,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
        total_weight = float(weights.sum())
        curves: dict[str, list[dict[str, Any]]] = {}
        ece_by_class: dict[str, float] = {}
        for class_idx, class_label in enumerate(self.classes_):
            scores = probabilities[:, class_idx]
            actual = (y_codes == class_idx).astype(np.float64)
            bin_codes = np.minimum((scores * n_bins).astype(np.int64), n_bins - 1)
            rows: list[dict[str, Any]] = []
            ece = 0.0
            for bin_idx in range(n_bins):
                mask = bin_codes == bin_idx
                bin_weight = float(weights[mask].sum())
                if bin_weight <= 0.0:
                    continue
                predicted_mean = float(np.sum(weights[mask] * scores[mask]) / bin_weight)
                actual_rate = float(np.sum(weights[mask] * actual[mask]) / bin_weight)
                ece += (bin_weight / total_weight) * abs(actual_rate - predicted_mean)
                rows.append(
                    {
                        "bin": bin_idx,
                        "lower": bin_idx / n_bins,
                        "upper": (bin_idx + 1) / n_bins,
                        "n_rows": int(np.sum(mask)),
                        "weight": bin_weight,
                        "predicted_mean": predicted_mean,
                        "actual_rate": actual_rate,
                        "error": actual_rate - predicted_mean,
                    }
                )
            curves[class_label] = rows
            ece_by_class[class_label] = float(ece)
        return curves, ece_by_class

    def _multiclass_ece_and_reliability(
        self,
        probabilities: np.ndarray,
        y_codes: np.ndarray,
        weights: np.ndarray,
        *,
        n_bins: int = 10,
    ) -> tuple[float, dict[str, dict[str, Any]]]:
        total_weight = float(weights.sum())
        predicted_codes = np.argmax(probabilities, axis=1)
        confidence = probabilities[np.arange(probabilities.shape[0]), predicted_codes]
        correct = (predicted_codes == y_codes).astype(np.float64)

        bin_codes = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)
        ece = 0.0
        for bin_idx in range(n_bins):
            mask = bin_codes == bin_idx
            bin_weight = float(weights[mask].sum())
            if bin_weight <= 0.0:
                continue
            accuracy = float(np.sum(weights[mask] * correct[mask]) / bin_weight)
            mean_confidence = float(np.sum(weights[mask] * confidence[mask]) / bin_weight)
            ece += (bin_weight / total_weight) * abs(accuracy - mean_confidence)

        reliability: dict[str, dict[str, Any]] = {}
        for class_idx, class_label in enumerate(self.classes_):
            mask = predicted_codes == class_idx
            class_weight = float(weights[mask].sum())
            if class_weight <= 0.0:
                reliability[class_label] = {
                    "n_rows": 0,
                    "weight": 0.0,
                    "accuracy": None,
                    "mean_confidence": None,
                }
                continue
            reliability[class_label] = {
                "n_rows": int(np.sum(mask)),
                "weight": class_weight,
                "accuracy": float(np.sum(weights[mask] * correct[mask]) / class_weight),
                "mean_confidence": float(np.sum(weights[mask] * confidence[mask]) / class_weight),
            }
        return float(ece), reliability

    def _dataset_diagnostics_from_arrays(
        self,
        *,
        name: str,
        probabilities: np.ndarray,
        y_codes: np.ndarray,
        weights: np.ndarray,
    ) -> MultinomialDatasetDiagnostics:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        y_codes = np.asarray(y_codes, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if y_codes.shape[0] != probabilities.shape[0]:
            raise PredictionError("Cannot compute diagnostics: response length mismatch.")
        if weights.shape[0] != probabilities.shape[0]:
            raise PredictionError("Cannot compute diagnostics: weight length mismatch.")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValidationError("weights must be finite and non-negative.")
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            raise PredictionError("Cannot compute diagnostics: fitted weights have zero total.")

        row_idx = np.arange(probabilities.shape[0])
        observed_prob = np.clip(probabilities[row_idx, y_codes], np.finfo(float).tiny, 1.0)
        log_loss = float(-np.sum(weights * np.log(observed_prob)) / total_weight)

        predicted_codes = np.argmax(probabilities, axis=1)
        accuracy = float(np.sum(weights * (predicted_codes == y_codes)) / total_weight)
        top_k = min(2, len(self.classes_))
        top_order = np.argsort(-probabilities, axis=1)[:, :top_k]
        top_2_hit = np.any(top_order == y_codes[:, None], axis=1)
        top_2_accuracy = float(np.sum(weights * top_2_hit) / total_weight)

        k = len(self.classes_)
        confusion = np.zeros((k, k), dtype=np.float64)
        np.add.at(confusion, (y_codes, predicted_codes), weights)

        actual_counts = np.bincount(y_codes, weights=weights, minlength=k).astype(np.float64)
        predicted_counts = (probabilities * weights[:, None]).sum(axis=0)
        actual_mix = actual_counts / total_weight
        predicted_mix = predicted_counts / total_weight

        per_class_metrics: dict[str, dict[str, float | None]] = {}
        precision_values: list[float | None] = []
        recall_values: list[float | None] = []
        f1_values: list[float | None] = []
        predicted_hard_counts = confusion.sum(axis=0)
        actual_hard_counts = confusion.sum(axis=1)
        for class_idx, class_label in enumerate(self.classes_):
            true_positive = float(confusion[class_idx, class_idx])
            precision = _optional_ratio(true_positive, float(predicted_hard_counts[class_idx]))
            recall = _optional_ratio(true_positive, float(actual_hard_counts[class_idx]))
            f1 = _optional_f1(precision, recall)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            per_class_metrics[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "actual_weight": float(actual_hard_counts[class_idx]),
                "predicted_hard_weight": float(predicted_hard_counts[class_idx]),
                "true_positive_weight": true_positive,
            }

        class_calibration, ece_by_class = self._calibration_by_class(
            probabilities, y_codes, weights
        )
        multiclass_ece, reliability = self._multiclass_ece_and_reliability(
            probabilities, y_codes, weights
        )

        actual_class_mix = {
            label: float(actual_mix[idx]) for idx, label in enumerate(self.classes_)
        }
        actual_class_counts = {
            label: float(actual_counts[idx]) for idx, label in enumerate(self.classes_)
        }
        predicted_class_counts = {
            label: float(predicted_counts[idx]) for idx, label in enumerate(self.classes_)
        }
        predicted_class_mix = {
            label: float(predicted_mix[idx]) for idx, label in enumerate(self.classes_)
        }
        class_mix_error = {
            label: float(actual_mix[idx] - predicted_mix[idx])
            for idx, label in enumerate(self.classes_)
        }

        return MultinomialDatasetDiagnostics(
            name=name,
            classes=list(self.classes_),
            nobs=int(probabilities.shape[0]),
            total_weight=total_weight,
            log_loss=log_loss,
            deviance=float(-2.0 * np.sum(weights * np.log(observed_prob))),
            accuracy=accuracy,
            top_2_accuracy=top_2_accuracy,
            balanced_accuracy=_mean_defined(recall_values),
            macro_precision=_mean_defined(precision_values),
            macro_recall=_mean_defined(recall_values),
            macro_f1=_mean_defined(f1_values),
            confusion_matrix=confusion,
            actual_class_counts=actual_class_counts,
            predicted_class_counts=predicted_class_counts,
            actual_class_mix=actual_class_mix,
            predicted_class_mix=predicted_class_mix,
            class_mix_error=class_mix_error,
            per_class_metrics=per_class_metrics,
            class_calibration=class_calibration,
            expected_calibration_error_by_class=ece_by_class,
            multiclass_expected_calibration_error=multiclass_ece,
            reliability_by_winning_class=reliability,
        )

    def _factor_diagnostics(
        self,
        *,
        dataset_name: str,
        data: Any,
        y_codes: np.ndarray,
        probabilities: np.ndarray,
        weights: np.ndarray,
        categorical_factors: list[str],
        continuous_factors: list[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        factor_specs = [(factor, "categorical") for factor in categorical_factors]
        factor_specs.extend((factor, "continuous") for factor in continuous_factors)
        for factor, factor_type in factor_specs:
            if factor not in data.columns:
                raise PredictionError(f"diagnostic factor {factor!r} is not present in data.")
            if factor_type == "continuous":
                labels = _continuous_factor_bins(data[factor].to_numpy())
            else:
                labels = _stringify_factor_values(data[factor].to_list())

            for level in sorted(set(labels.tolist())):
                mask = labels == level
                level_weight = float(weights[mask].sum())
                if level_weight <= 0.0:
                    continue
                actual_counts = np.bincount(
                    y_codes[mask], weights=weights[mask], minlength=len(self.classes_)
                ).astype(np.float64)
                predicted_counts = (probabilities[mask] * weights[mask, None]).sum(axis=0)
                actual_mix = actual_counts / level_weight
                predicted_mix = predicted_counts / level_weight
                expected_positive = predicted_counts > 0.0
                chi_square = (
                    float(
                        np.sum(
                            (actual_counts[expected_positive] - predicted_counts[expected_positive])
                            ** 2
                            / predicted_counts[expected_positive]
                        )
                    )
                    if np.any(expected_positive)
                    else None
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "factor": factor,
                        "factor_type": factor_type,
                        "level": str(level),
                        "n_rows": int(np.sum(mask)),
                        "weight": level_weight,
                        "actual_class_mix": {
                            label: float(actual_mix[idx]) for idx, label in enumerate(self.classes_)
                        },
                        "predicted_class_mix": {
                            label: float(predicted_mix[idx])
                            for idx, label in enumerate(self.classes_)
                        },
                        "class_mix_error": {
                            label: float(actual_mix[idx] - predicted_mix[idx])
                            for idx, label in enumerate(self.classes_)
                        },
                        "observed_winning_class": self.classes_[int(np.argmax(actual_counts))],
                        "predicted_winning_class": self.classes_[int(np.argmax(predicted_counts))],
                        "chi_square_class_mix": chi_square,
                    }
                )
        return rows

    def _train_test_comparison(
        self,
        train: MultinomialDatasetDiagnostics,
        test: MultinomialDatasetDiagnostics,
    ) -> dict[str, float | None]:
        train_mae = float(np.mean([abs(value) for value in train.class_mix_error.values()]))
        test_mae = float(np.mean([abs(value) for value in test.class_mix_error.values()]))

        def delta(test_value: float | None, train_value: float | None) -> float | None:
            if test_value is None or train_value is None:
                return None
            return float(test_value - train_value)

        return {
            "log_loss_delta": test.log_loss - train.log_loss,
            "accuracy_delta": test.accuracy - train.accuracy,
            "top_2_accuracy_delta": test.top_2_accuracy - train.top_2_accuracy,
            "balanced_accuracy_delta": delta(test.balanced_accuracy, train.balanced_accuracy),
            "macro_f1_delta": delta(test.macro_f1, train.macro_f1),
            "class_mix_mae_train": train_mae,
            "class_mix_mae_test": test_mae,
            "class_mix_mae_delta": test_mae - train_mae,
        }

    def diagnostics(
        self,
        train_data: Any | None = None,
        test_data: Any | None = None,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
        *,
        weights: str | None = None,
    ) -> MultinomialDiagnostics:
        categorical_factors = list(categorical_factors or [])
        continuous_factors = list(continuous_factors or [])
        factor_diagnostics: list[dict[str, Any]] = []

        using_fitted_training = train_data is None
        if using_fitted_training:
            if categorical_factors or continuous_factors:
                raise ValidationError("factor diagnostics require train_data.")
            # Use the unfolded observation weights so the reported class mix
            # reflects the data distribution, not the class-reweighted training
            # objective (consistent with the supplied-data path below).
            n_train = len(np.asarray(self._result.y_codes))
            fitted_weights = (
                self._fit_row_weights
                if self._fit_row_weights is not None
                else np.ones(n_train, dtype=np.float64)
            )
            train = self._dataset_diagnostics_from_arrays(
                name="train",
                probabilities=self.fitted_probabilities,
                y_codes=np.asarray(self._result.y_codes, dtype=np.int64),
                weights=np.asarray(fitted_weights, dtype=np.float64),
            )
        else:
            train_frame, train_probabilities, train_y, train_weights = (
                self._diagnostic_arrays_from_data(
                    train_data,
                    categorical_factors=categorical_factors,
                    continuous_factors=continuous_factors,
                    weights_override=weights,
                )
            )
            train = self._dataset_diagnostics_from_arrays(
                name="train",
                probabilities=train_probabilities,
                y_codes=train_y,
                weights=train_weights,
            )
            factor_diagnostics.extend(
                self._factor_diagnostics(
                    dataset_name="train",
                    data=train_frame,
                    y_codes=train_y,
                    probabilities=train_probabilities,
                    weights=train_weights,
                    categorical_factors=categorical_factors,
                    continuous_factors=continuous_factors,
                )
            )

        test = None
        if test_data is not None:
            test_frame, test_probabilities, test_y, test_weights = (
                self._diagnostic_arrays_from_data(
                    test_data,
                    categorical_factors=categorical_factors,
                    continuous_factors=continuous_factors,
                    weights_override=weights,
                )
            )
            test = self._dataset_diagnostics_from_arrays(
                name="test",
                probabilities=test_probabilities,
                y_codes=test_y,
                weights=test_weights,
            )
            factor_diagnostics.extend(
                self._factor_diagnostics(
                    dataset_name="test",
                    data=test_frame,
                    y_codes=test_y,
                    probabilities=test_probabilities,
                    weights=test_weights,
                    categorical_factors=categorical_factors,
                    continuous_factors=continuous_factors,
                )
            )

        train_test_comparison = None if test is None else self._train_test_comparison(train, test)
        fit_aic = self.aic()
        fit_bic = self.bic()
        if using_fitted_training:
            aic = fit_aic
            bic = fit_bic
        else:
            aic = None if fit_aic is None else train.deviance + 2.0 * self.n_params
            bic = (
                None
                if fit_bic is None
                else train.deviance + self.n_params * math.log(max(train.nobs, 1))
            )
        # McFadden's pseudo-R2 must pair the model deviance with an intercept-only
        # null on the SAME rows. The no-train_data path uses the fit-time null
        # (which the solver computed honouring availability/offset); the
        # supplied-data path derives a class-share null on the supplied rows so
        # the ratio is not a cross-basis number.
        if using_fitted_training:
            null_for_r2 = self.null_deviance
        else:
            null_for_r2 = _share_based_null_deviance(train.actual_class_counts, train.total_weight)
        pseudo_r2 = None
        if null_for_r2 is not None and null_for_r2 > 0.0:
            pseudo_r2 = 1.0 - train.deviance / null_for_r2

        return MultinomialDiagnostics(
            model_type="baseline-category multinomial logit",
            classes=list(self.classes_),
            reference=self.reference_,
            nobs=train.nobs,
            n_params=self.n_params,
            log_likelihood=-0.5 * train.deviance,
            deviance=train.deviance,
            null_deviance=self.null_deviance,
            mcfadden_pseudo_r2=pseudo_r2,
            aic=aic,
            bic=bic,
            log_loss=train.log_loss,
            accuracy=train.accuracy,
            top_2_accuracy=train.top_2_accuracy,
            balanced_accuracy=train.balanced_accuracy,
            macro_precision=train.macro_precision,
            macro_recall=train.macro_recall,
            macro_f1=train.macro_f1,
            confusion_matrix=train.confusion_matrix,
            actual_class_counts=train.actual_class_counts,
            predicted_class_counts=train.predicted_class_counts,
            actual_class_mix=train.actual_class_mix,
            predicted_class_mix=train.predicted_class_mix,
            class_mix_error=train.class_mix_error,
            per_class_metrics=train.per_class_metrics,
            class_calibration=train.class_calibration,
            expected_calibration_error_by_class=train.expected_calibration_error_by_class,
            multiclass_expected_calibration_error=train.multiclass_expected_calibration_error,
            reliability_by_winning_class=train.reliability_by_winning_class,
            factor_diagnostics=factor_diagnostics,
            train=train,
            test=test,
            train_test_comparison=train_test_comparison,
            inference_status=self.inference_status,
            solver_status=self.solver_status,
            converged=self.converged,
            warnings=self.warnings,
        )

    def diagnostics_json(
        self,
        train_data: Any | None = None,
        test_data: Any | None = None,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        return self.diagnostics(
            train_data=train_data,
            test_data=test_data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
        ).to_json(indent=indent)

    def relevel(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ValidationError(
            "relevel() is not defined for multinomial models. Use diagnostics to assess "
            "class-mix calibration, or fit_calibration(method='intercept') for vector "
            "intercept calibration."
        )

    def _calibration_shift_vector(
        self, calibration: MultinomialInterceptCalibration | None
    ) -> np.ndarray | None:
        if calibration is None:
            return None
        if not isinstance(calibration, MultinomialInterceptCalibration):
            raise ValidationError("calibration must be a MultinomialInterceptCalibration.")
        if calibration.classes != self.classes_:
            raise ValidationError("calibration classes must match the model classes and ordering.")
        if calibration.reference != self.reference_:
            raise ValidationError("calibration reference must match the model reference.")
        return calibration.shift_vector

    def _calibration_weights(
        self,
        data: Any,
        weights: str | np.ndarray | None,
    ) -> np.ndarray:
        weights_to_use = self._weights_spec if weights is None and self._weights_spec else weights
        if weights_to_use is None:
            resolved = np.ones(len(data), dtype=np.float64)
        elif isinstance(weights_to_use, str):
            if weights_to_use not in data.columns:
                raise PredictionError(f"weights column {weights_to_use!r} is not present in data.")
            resolved = _as_float_array(
                data[weights_to_use].to_numpy(), name="weights", length=len(data)
            )
        else:
            resolved = _as_float_array(weights_to_use, name="weights", length=len(data))
        if np.any(resolved < 0.0):
            raise ValidationError("calibration weights must be finite and non-negative.")
        if float(np.sum(resolved)) <= 0.0:
            raise ValidationError("calibration weights must have positive total.")
        return resolved

    def fit_calibration(
        self,
        data: Any,
        *,
        method: str = "intercept",
        weights: str | np.ndarray | None = None,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> MultinomialInterceptCalibration:
        """Fit a standalone multinomial calibration object.

        Phase 6 supports vector-intercept calibration only. It shifts class
        logits so weighted predicted class mix matches the calibration data,
        without mutating the fitted model coefficients.

        Prefer a held-out calibration fold or out-of-fold predictions: fitting
        calibration on the training rows overstates calibration quality, and
        vector-intercept calibration only corrects global class mix rather than
        segment-varying miscalibration.
        """

        normalized_method = str(method).replace("-", "_").lower()
        if normalized_method not in {"intercept", "vector_intercept"}:
            raise ValidationError(
                "fit_calibration currently supports method='intercept' only; "
                "temperature, isotonic, and segment-level calibration are not yet implemented."
            )
        if self.response is None:
            raise PredictionError(
                "Cannot fit calibration: this model has no stored response column metadata."
            )
        if max_iter <= 0:
            raise ValidationError("max_iter must be positive.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValidationError("tol must be positive and finite.")

        extra_columns = {self.response}
        weights_to_use = self._weights_spec if weights is None and self._weights_spec else weights
        if isinstance(weights_to_use, str):
            extra_columns.add(weights_to_use)
        calibration_data = self._prepare_prediction_data(
            data, availability, offset, extra_columns=extra_columns
        )
        y_codes = self._class_codes_from_values(
            calibration_data[self.response].to_list(), name=self.response
        )
        resolved_weights = self._calibration_weights(calibration_data, weights)
        availability_matrix = self._resolve_prediction_availability(calibration_data, availability)
        if not np.all(availability_matrix[np.arange(len(y_codes)), y_codes]):
            raise ValidationError("calibration data contains an observed class marked unavailable.")

        class_counts = np.bincount(
            y_codes, weights=resolved_weights, minlength=len(self.classes_)
        ).astype(np.float64)
        missing = [
            class_label
            for class_label, class_weight in zip(self.classes_, class_counts, strict=True)
            if class_weight <= 0.0
        ]
        if missing:
            raise ValidationError(
                "vector intercept calibration requires positive weighted observations "
                f"for every class; missing classes: {missing}."
            )

        logits = self.decision_function(calibration_data, availability=availability, offset=offset)
        if logits.shape != availability_matrix.shape:
            raise PredictionError("calibration logits and availability shapes do not match.")

        row_idx = np.arange(len(y_codes))
        non_reference = [
            idx for idx, class_label in enumerate(self.classes_) if class_label != self.reference_
        ]
        shifts = np.zeros(len(self.classes_), dtype=np.float64)
        total_weight = float(np.sum(resolved_weights))

        def objective(candidate_shifts: np.ndarray) -> float:
            shifted = logits + candidate_shifts[None, :]
            masked = np.where(availability_matrix, shifted, -np.inf)
            max_eta = np.max(masked, axis=1)
            if not np.all(np.isfinite(max_eta)):
                raise PredictionError(
                    "availability leaves at least one calibration row with no classes."
                )
            exp_eta = np.where(availability_matrix, np.exp(masked - max_eta[:, None]), 0.0)
            log_denom = max_eta + np.log(exp_eta.sum(axis=1))
            return float(np.sum(resolved_weights * (log_denom - shifted[row_idx, y_codes])))

        current_objective = objective(shifts)
        converged = False
        warnings: list[str] = []
        iterations = 0
        for iteration in range(max_iter):
            iterations = iteration
            probabilities = _masked_softmax(logits + shifts[None, :], availability_matrix)
            actual = np.zeros_like(probabilities)
            actual[row_idx, y_codes] = 1.0
            gradient = (
                resolved_weights[:, None]
                * (probabilities[:, non_reference] - actual[:, non_reference])
            ).sum(axis=0)
            gradient_norm = float(np.max(np.abs(gradient)) / total_weight)
            if gradient_norm <= tol:
                converged = True
                break

            p_non_reference = probabilities[:, non_reference]
            hessian = -(p_non_reference * resolved_weights[:, None]).T @ p_non_reference
            hessian += np.diag((resolved_weights[:, None] * p_non_reference).sum(axis=0))
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as exc:
                raise ValidationError(
                    "Could not fit vector intercept calibration because the calibration "
                    "Hessian is singular. This usually means the calibration sample has "
                    "disconnected availability patterns or too little support by class."
                ) from exc

            accepted = False
            step_scale = 1.0
            for _ in range(40):
                candidate = shifts.copy()
                candidate[non_reference] -= step_scale * step
                candidate[self.reference_index_] = 0.0
                candidate_objective = objective(candidate)
                if candidate_objective <= current_objective + 1e-12:
                    shifts = candidate
                    current_objective = candidate_objective
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                warnings.append(
                    "vector intercept calibration stopped because the Newton step "
                    "could not improve the likelihood."
                )
                break

        if not converged:
            probabilities = _masked_softmax(logits + shifts[None, :], availability_matrix)
            actual = np.zeros_like(probabilities)
            actual[row_idx, y_codes] = 1.0
            gradient = (
                resolved_weights[:, None]
                * (probabilities[:, non_reference] - actual[:, non_reference])
            ).sum(axis=0)
            gradient_norm = float(np.max(np.abs(gradient)) / total_weight)
            if gradient_norm <= tol:
                converged = True
            else:
                warnings.append(
                    "vector intercept calibration did not converge within "
                    f"{max_iter} iterations; max normalized gradient={gradient_norm:.3e}."
                )

        base_probabilities = _masked_softmax(logits, availability_matrix)
        calibrated_probabilities = _masked_softmax(logits + shifts[None, :], availability_matrix)
        actual_mix = class_counts / total_weight
        base_mix = _weighted_probability_mix(base_probabilities, resolved_weights)
        calibrated_mix = _weighted_probability_mix(calibrated_probabilities, resolved_weights)

        return MultinomialInterceptCalibration(
            classes=list(self.classes_),
            reference=self.reference_,
            shifts={label: float(shifts[idx]) for idx, label in enumerate(self.classes_)},
            actual_class_mix={
                label: float(actual_mix[idx]) for idx, label in enumerate(self.classes_)
            },
            base_predicted_class_mix={
                label: float(base_mix[idx]) for idx, label in enumerate(self.classes_)
            },
            calibrated_class_mix={
                label: float(calibrated_mix[idx]) for idx, label in enumerate(self.classes_)
            },
            nobs=len(calibration_data),
            total_weight=total_weight,
            iterations=iterations,
            converged=converged,
            warnings=warnings,
        )

    def _prepare_prediction_data(
        self,
        data: Any,
        availability: Any,
        offset: Any,
        extra_columns: set[str] | None = None,
    ) -> Any:
        if self._builder is None:
            raise PredictionError("Cannot predict: this model has no stored design builder.")
        availability_to_use = availability if availability is not None else self._availability_spec
        offset_to_use = offset if offset is not None else self._offset_spec
        needed = _extract_needed_columns(
            terms=self._terms_dict,
            interactions=self._interactions_spec,
            input_transforms=self._input_transforms,
        )
        needed |= _target_encoding_source_columns(
            self._terms_dict,
            self._interactions_spec,
            self._input_transforms,
        )
        needed |= _extra_needed_columns(
            [availability_to_use, offset_to_use], self._input_transforms
        )
        needed |= _alternative_needed_columns(self._alternative_terms_spec, self._input_transforms)
        needed |= set(extra_columns or set())
        data = _collect_lazyframe(data, needed)
        if self._compiled_input_transforms:
            drop_outputs = [
                spec["output"] for spec in self._input_transforms if spec["output"] in data.columns
            ]
            if drop_outputs:
                data = data.drop(drop_outputs)
            data = apply_input_transforms(data, self._compiled_input_transforms)
        return data

    def _prediction_alternative_arrays(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        generic, specific, _generic_names, _specific_names = _resolve_alternative_arrays(
            data, self.classes_, self._alternative_terms_spec
        )
        if self._target_encoding_state is not None:
            te_specific = self._target_encoding_state.transform(data)
            specific = _append_alternative_specific_terms(specific, te_specific)
        return generic, specific

    def _resolve_prediction_availability(self, data: Any, availability: Any) -> np.ndarray:
        spec = availability if availability is not None else self._availability_spec
        allow_arrays = availability is not None
        if availability is None and self._array_availability_requires_prediction_override:
            raise PredictionError(
                "This model was fit with array availability, which is fit-time data. "
                "Pass availability= for prediction data, or fit with column names/scalars."
            )
        return _resolve_class_matrix(
            data,
            spec,
            self.classes_,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=allow_arrays,
        )

    def _resolve_prediction_offset(self, data: Any, offset: Any) -> np.ndarray:
        spec = offset if offset is not None else self._offset_spec
        allow_arrays = offset is not None
        if offset is None and self._array_offset_requires_prediction_override:
            raise PredictionError(
                "This model was fit with array offsets, which are fit-time data. "
                "Pass offset= for prediction data, or fit with column names/scalars."
            )
        return _resolve_class_matrix(
            data,
            spec,
            self.classes_,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=allow_arrays,
        )

    def decision_function(
        self,
        new_data: Any,
        *,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        include_reference: bool = True,
    ) -> np.ndarray:
        data = self._prepare_prediction_data(new_data, availability, offset)
        n_rows = len(data)
        n_features = len(self.feature_names)
        chunk_size = _compute_predict_chunk_size(n_features)
        logits = np.empty((n_rows, len(self.classes_)), dtype=np.float64)
        params = self.params
        alternative_generic_coefficients = self.alternative_generic_coefficients
        alternative_specific_coefficients = self.alternative_specific_coefficients

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = data.slice(start, stop - start)
            x_chunk = self._builder.transform_new_data(chunk)
            offset_chunk = self._resolve_prediction_offset(
                chunk,
                _slice_class_matrix_override(
                    offset,
                    start=start,
                    stop=stop,
                    n_rows=n_rows,
                    n_classes=len(self.classes_),
                    name="offset",
                ),
            )
            logits_chunk = offset_chunk.copy()
            alternative_generic, alternative_specific = self._prediction_alternative_arrays(chunk)
            if alternative_generic_coefficients.size:
                logits_chunk += np.tensordot(
                    alternative_generic, alternative_generic_coefficients, axes=([2], [0])
                )
            block = 0
            for class_idx, class_label in enumerate(self.classes_):
                if class_label == self.reference_:
                    continue
                logits_chunk[:, class_idx] += x_chunk @ params[block, :]
                if alternative_specific_coefficients.size:
                    logits_chunk[:, class_idx] += (
                        alternative_specific[:, class_idx, :]
                        @ alternative_specific_coefficients[block, :]
                    )
                block += 1
            logits[start:stop, :] = logits_chunk

        if include_reference:
            return logits
        keep = [idx for idx, label in enumerate(self.classes_) if label != self.reference_]
        return logits[:, keep]

    def predict_proba(
        self,
        new_data: Any,
        *,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
        return_format: str = "numpy",
    ) -> np.ndarray:
        data = self._prepare_prediction_data(new_data, availability, offset)
        logits = self.decision_function(data, availability=availability, offset=offset)
        calibration_shift = self._calibration_shift_vector(calibration)
        if calibration_shift is not None:
            logits = logits + calibration_shift[None, :]
        availability_matrix = self._resolve_prediction_availability(data, availability)
        probabilities = _masked_softmax(logits, availability_matrix)
        if return_format == "numpy":
            return probabilities
        if return_format == "polars":
            import polars as pl

            return pl.DataFrame(
                {f"prob_{label}": probabilities[:, idx] for idx, label in enumerate(self.classes_)}
            )
        raise ValidationError("return_format must be 'numpy' or 'polars'.")

    def predict_log_proba(
        self,
        new_data: Any,
        *,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
        return_format: str = "numpy",
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            new_data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        with np.errstate(divide="ignore"):
            log_probabilities = np.log(probabilities)
        if return_format == "numpy":
            return log_probabilities
        if return_format == "polars":
            import polars as pl

            return pl.DataFrame(
                {
                    f"log_prob_{label}": log_probabilities[:, idx]
                    for idx, label in enumerate(self.classes_)
                }
            )
        raise ValidationError("return_format must be 'numpy' or 'polars'.")

    def predict(
        self,
        new_data: Any,
        *,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            new_data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        labels = np.asarray(self.classes_, dtype=object)
        return labels[np.argmax(probabilities, axis=1)]

    def predict_top_k(
        self,
        new_data: Any,
        *,
        k: int = 2,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
    ) -> Any:
        if k <= 0 or k > len(self.classes_):
            raise ValidationError(f"k must be in [1, {len(self.classes_)}].")
        probabilities = self.predict_proba(
            new_data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        order = np.argsort(-probabilities, axis=1)[:, :k]
        import polars as pl

        data: dict[str, Any] = {}
        for rank in range(k):
            idx = order[:, rank]
            data[f"class_{rank + 1}"] = [self.classes_[class_idx] for class_idx in idx]
            data[f"prob_{rank + 1}"] = probabilities[np.arange(probabilities.shape[0]), idx]
        return pl.DataFrame(data)

    def tier_mix(
        self,
        new_data: Any,
        *,
        weights: str | np.ndarray | None = None,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
        return_format: str = "dict",
    ) -> dict[str, float]:
        extra_columns = {weights} if isinstance(weights, str) else None
        data = self._prepare_prediction_data(
            new_data, availability, offset, extra_columns=extra_columns
        )
        probabilities = self.predict_proba(
            data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        if weights is None:
            mix = probabilities.mean(axis=0)
        else:
            if isinstance(weights, str):
                if weights not in data.columns:
                    raise PredictionError(f"weights column {weights!r} is not present in data.")
                w = data[weights].to_numpy().astype(np.float64)
            else:
                w = _as_float_array(weights, name="weights", length=len(data))
            total = float(np.sum(w))
            if total <= 0.0:
                raise ValidationError("tier_mix weights must have positive total.")
            mix = (probabilities * w[:, None]).sum(axis=0) / total

        if return_format == "dict":
            return {label: float(mix[idx]) for idx, label in enumerate(self.classes_)}
        if return_format == "polars":
            import polars as pl

            return pl.DataFrame({"class": self.classes_, "probability": mix})
        raise ValidationError("return_format must be 'dict' or 'polars'.")

    def _scenario_weights(self, data: Any, weights: str | np.ndarray | None) -> np.ndarray:
        if weights is None:
            return np.ones(len(data), dtype=np.float64)
        if isinstance(weights, str):
            if weights not in data.columns:
                raise PredictionError(f"weights column {weights!r} is not present in data.")
            resolved = _as_float_array(data[weights].to_numpy(), name="weights", length=len(data))
        else:
            resolved = _as_float_array(weights, name="weights", length=len(data))
        if np.any(resolved < 0.0):
            raise ValidationError("weights must be non-negative.")
        if float(np.sum(resolved)) <= 0.0:
            raise ValidationError("scenario weights must have positive total.")
        return resolved

    def _weighted_class_mix(self, probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return (probabilities * weights[:, None]).sum(axis=0) / float(np.sum(weights))

    def _scenario_expected_value(
        self,
        base_data: Any,
        scenario_data: Any,
        base_probabilities: np.ndarray,
        scenario_probabilities: np.ndarray,
        weights: np.ndarray,
        value_columns: dict[str, str] | None,
    ) -> dict[str, float] | None:
        if value_columns is None:
            return None
        base_values = np.zeros_like(base_probabilities)
        scenario_values = np.zeros_like(scenario_probabilities)
        for class_idx, class_label in enumerate(self.classes_):
            column = value_columns.get(class_label)
            if column is None:
                continue
            if column not in base_data.columns:
                raise PredictionError(f"value column {column!r} is not present in data.")
            if column not in scenario_data.columns:
                raise PredictionError(f"value column {column!r} is not present in scenario data.")
            base_values[:, class_idx] = _as_float_array(
                base_data[column].to_numpy(),
                name=f"value_columns[{class_label!r}]",
                length=len(base_data),
            )
            scenario_values[:, class_idx] = _as_float_array(
                scenario_data[column].to_numpy(),
                name=f"value_columns[{class_label!r}]",
                length=len(scenario_data),
            )
        total_weight = float(np.sum(weights))
        base = float(
            np.sum(weights * np.sum(base_probabilities * base_values, axis=1)) / total_weight
        )
        scenario = float(
            np.sum(weights * np.sum(scenario_probabilities * scenario_values, axis=1))
            / total_weight
        )
        return {"base": base, "scenario": scenario, "delta": scenario - base}

    def _scenario_segment_mix(
        self,
        data: Any,
        base_probabilities: np.ndarray,
        scenario_probabilities: np.ndarray,
        weights: np.ndarray,
        categorical_factors: list[str],
        continuous_factors: list[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        factor_specs = [(factor, "categorical") for factor in categorical_factors]
        factor_specs.extend((factor, "continuous") for factor in continuous_factors)
        for factor, factor_type in factor_specs:
            if factor not in data.columns:
                raise PredictionError(f"scenario factor {factor!r} is not present in data.")
            labels = (
                _continuous_factor_bins(data[factor].to_numpy())
                if factor_type == "continuous"
                else _stringify_factor_values(data[factor].to_list())
            )
            for level in sorted(set(labels.tolist())):
                mask = labels == level
                level_weight = float(weights[mask].sum())
                if level_weight <= 0.0:
                    continue
                base_mix = self._weighted_class_mix(base_probabilities[mask], weights[mask])
                scenario_mix = self._weighted_class_mix(scenario_probabilities[mask], weights[mask])
                rows.append(
                    {
                        "factor": factor,
                        "factor_type": factor_type,
                        "level": str(level),
                        "n_rows": int(np.sum(mask)),
                        "weight": level_weight,
                        "base_class_mix": {
                            label: float(base_mix[idx]) for idx, label in enumerate(self.classes_)
                        },
                        "scenario_class_mix": {
                            label: float(scenario_mix[idx])
                            for idx, label in enumerate(self.classes_)
                        },
                        "class_mix_delta": {
                            label: float(scenario_mix[idx] - base_mix[idx])
                            for idx, label in enumerate(self.classes_)
                        },
                    }
                )
        return rows

    def scenario(
        self,
        new_data: Any,
        changes: dict[str, float | np.ndarray],
        *,
        weights: str | np.ndarray | None = None,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
        calibration: MultinomialInterceptCalibration | None = None,
        value_columns: dict[str, str] | None = None,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
    ) -> MultinomialScenario:
        if not self._alternative_terms_spec:
            raise ValidationError("scenario() requires a model fit with alternative_terms.")
        if not isinstance(changes, dict) or not changes:
            raise ValidationError("changes must be a non-empty dict of column changes.")
        alternative_columns = _alternative_columns_raw(self._alternative_terms_spec)
        unknown = sorted(set(changes) - alternative_columns)
        if unknown:
            raise ValidationError(
                f"scenario changes reference columns not used by alternative_terms: {unknown}."
            )

        categorical_factors = list(categorical_factors or [])
        continuous_factors = list(continuous_factors or [])
        extra_columns = set(changes)
        extra_columns.update(categorical_factors)
        extra_columns.update(continuous_factors)
        if isinstance(weights, str):
            extra_columns.add(weights)
        if value_columns is not None:
            extra_columns.update(value_columns.values())

        data = self._prepare_prediction_data(
            new_data, availability, offset, extra_columns=extra_columns
        )
        import polars as pl

        scenario_data = data
        for column, change in changes.items():
            if column not in scenario_data.columns:
                raise PredictionError(f"scenario change column {column!r} is not present in data.")
            if np.isscalar(change):
                multiplier = float(change)
                if not np.isfinite(multiplier):
                    raise ValidationError(f"scenario multiplier for {column!r} must be finite.")
                scenario_data = scenario_data.with_columns(
                    (pl.col(column) * multiplier).alias(column)
                )
            else:
                values = _as_float_array(change, name=f"changes[{column!r}]", length=len(data))
                scenario_data = scenario_data.with_columns(pl.Series(column, values))

        base_probabilities = self.predict_proba(
            data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        scenario_probabilities = self.predict_proba(
            scenario_data,
            availability=availability,
            offset=offset,
            calibration=calibration,
            return_format="numpy",
        )
        resolved_weights = self._scenario_weights(data, weights)
        base_mix = self._weighted_class_mix(base_probabilities, resolved_weights)
        scenario_mix = self._weighted_class_mix(scenario_probabilities, resolved_weights)

        return MultinomialScenario(
            classes=list(self.classes_),
            nobs=len(data),
            total_weight=float(np.sum(resolved_weights)),
            base_class_mix={label: float(base_mix[idx]) for idx, label in enumerate(self.classes_)},
            scenario_class_mix={
                label: float(scenario_mix[idx]) for idx, label in enumerate(self.classes_)
            },
            class_mix_delta={
                label: float(scenario_mix[idx] - base_mix[idx])
                for idx, label in enumerate(self.classes_)
            },
            base_probabilities=base_probabilities,
            scenario_probabilities=scenario_probabilities,
            expected_value=self._scenario_expected_value(
                data,
                scenario_data,
                base_probabilities,
                scenario_probabilities,
                resolved_weights,
                value_columns,
            ),
            segment_mix=self._scenario_segment_mix(
                data,
                base_probabilities,
                scenario_probabilities,
                resolved_weights,
                categorical_factors,
                continuous_factors,
            ),
        )

    def _covariance(self) -> np.ndarray | None:
        cov = getattr(self._result, "cov_params_unscaled", None)
        if cov is None:
            return None
        return np.asarray(cov, dtype=np.float64)

    def coef_table(self, return_format: str = "polars") -> Any:
        estimates = self.params
        alternative_generic = self.alternative_generic_coefficients
        alternative_specific = self.alternative_specific_coefficients
        cov = self._covariance()
        n_blocks, p = estimates.shape
        shared_size = estimates.size
        generic_size = alternative_generic.size
        if cov is not None:
            se_flat = np.sqrt(np.maximum(np.diag(cov), 0.0))
            se = se_flat[:shared_size].reshape(n_blocks, p)
            generic_se = se_flat[shared_size : shared_size + generic_size]
            specific_se = se_flat[shared_size + generic_size :].reshape(
                n_blocks, len(self.alternative_specific_feature_names)
            )
        else:
            se = np.full_like(estimates, np.nan)
            generic_se = np.full_like(alternative_generic, np.nan)
            specific_se = np.full_like(alternative_specific, np.nan)

        rows = []

        def append_row(
            *,
            class_label: str,
            feature: str,
            estimate: float,
            std_error: float,
            coefficient_type: str,
        ) -> None:
            if np.isfinite(std_error) and std_error > 0.0:
                z_value = estimate / std_error
                p_value = _normal_two_sided_p(z_value)
                ci_low = estimate - 1.959963984540054 * std_error
                ci_high = estimate + 1.959963984540054 * std_error
            else:
                z_value = p_value = ci_low = ci_high = float("nan")
            rows.append(
                {
                    "class": class_label,
                    "feature": feature,
                    "coefficient_type": coefficient_type,
                    "estimate": estimate,
                    "std_error": std_error,
                    "z": z_value,
                    "p_value": p_value,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "odds_ratio": math.exp(estimate) if np.isfinite(estimate) else np.nan,
                }
            )

        block = 0
        for class_label in self.classes_:
            if class_label == self.reference_:
                continue
            for feature_idx, feature in enumerate(self.feature_names):
                estimate = float(estimates[block, feature_idx])
                std_error = float(se[block, feature_idx])
                append_row(
                    class_label=class_label,
                    feature=feature,
                    estimate=estimate,
                    std_error=std_error,
                    coefficient_type="shared",
                )
            block += 1

        for term_idx, term_name in enumerate(self.alternative_generic_feature_names):
            append_row(
                class_label="all",
                feature=term_name,
                estimate=float(alternative_generic[term_idx]),
                std_error=float(generic_se[term_idx]),
                coefficient_type="alternative_generic",
            )

        block = 0
        for class_label in self.classes_:
            if class_label == self.reference_:
                continue
            for term_idx, term_name in enumerate(self.alternative_specific_feature_names):
                append_row(
                    class_label=class_label,
                    feature=term_name,
                    estimate=float(alternative_specific[block, term_idx]),
                    std_error=float(specific_se[block, term_idx]),
                    coefficient_type="alternative_class_specific",
                )
            block += 1

        if return_format == "records":
            return rows
        if return_format == "polars":
            import polars as pl

            return pl.DataFrame(rows)
        raise ValidationError("return_format must be 'polars' or 'records'.")

    def summary(self) -> str:
        aic = self.aic()
        bic = self.bic()
        diagnostics = self.diagnostics()

        def fmt(value: float | None, *, width: int = 10, precision: int = 4) -> str:
            if value is None or not np.isfinite(value):
                return f"{'NA':>{width}}"
            return f"{value:>{width}.{precision}f}"

        def fmt_p(value: float | None) -> str:
            if value is None or not np.isfinite(value):
                return f"{'NA':>8}"
            if value < 0.0001:
                return f"{'<0.0001':>8}"
            return f"{value:>8.4f}"

        if self._smooth_results:
            method = "Newton + Smooth"
        elif self.alpha <= 0.0:
            method = "Newton"
        elif self.l1_ratio >= 1.0:
            method = "Newton + Lasso"
        elif self.l1_ratio > 0.0:
            method = "Newton + Elastic Net"
        else:
            method = "Newton + Ridge"

        lines = [
            "=" * 78,
            "Multinomial Logit Results".center(78),
            "=" * 78,
            f"{'Classes:':<20} {', '.join(self.classes_)}",
            f"{'Reference:':<20} {self.reference_}",
            f"{'No. Observations:':<20} {self.nobs:>10}",
            f"{'Parameters:':<20} {self.n_params:>10}",
            f"{'Method:':<20} {method}",
            f"{'Iterations:':<20} {self.iterations:>10}",
            f"{'Converged:':<20} {self.converged!s:>10}",
            f"{'Solver status:':<20} {self.solver_status}",
            "",
            f"{'Log-Likelihood:':<20} {self.log_likelihood:>15.4f}",
            f"{'Deviance:':<20} {self.deviance:>15.4f}",
            f"{'Null Deviance:':<20} {self.null_deviance:>15.4f}",
            f"{'AIC:':<20} {'NA' if aic is None else f'{aic:.4f}':>15}",
            f"{'BIC:':<20} {'NA' if bic is None else f'{bic:.4f}':>15}",
            f"{'GCV:':<20} {'NA' if self.gcv is None else f'{self.gcv:.4f}':>15}",
            f"{'Total EDF:':<20} {'NA' if self.total_edf is None else f'{self.total_edf:.4f}':>15}",
            f"{'Log Loss:':<20} {diagnostics.log_loss:>15.4f}",
            f"{'Accuracy:':<20} {diagnostics.accuracy:>15.4f}",
            f"{'Top-2 Accuracy:':<20} {diagnostics.top_2_accuracy:>15.4f}",
            f"{'Inference:':<20} {self.inference_status}",
            "=" * 78,
        ]

        lines.extend(["", "Class Mix:", "-" * 78])
        lines.append(f"{'Class':<20} {'Actual':>12} {'Predicted':>12} {'Error':>12}")
        lines.append("-" * 78)
        for class_label in self.classes_:
            lines.append(
                f"{class_label:<20} "
                f"{diagnostics.actual_class_mix[class_label]:>12.4f} "
                f"{diagnostics.predicted_class_mix[class_label]:>12.4f} "
                f"{diagnostics.class_mix_error[class_label]:>12.4f}"
            )
        lines.append("-" * 78)

        if self._smooth_results:
            lines.extend(["", "Smooth Terms:", "-" * 78])
            lines.append(f"{'Variable':<20} {'Type':>8} {'k':>6} {'Lambda':>12} {'EDF':>10}")
            lines.append("-" * 78)
            for smooth in self._smooth_results:
                lines.append(
                    f"{smooth.variable:<20} {smooth.spline_type:>8} {smooth.k:>6} "
                    f"{smooth.lambda_:>12.4g} {smooth.edf:>10.4f}"
                )
            lines.append("-" * 78)

        coef_rows = self.coef_table(return_format="records")
        coefficient_type_labels = {
            "shared": "shared",
            "alternative_generic": "alt_gen",
            "alternative_class_specific": "alt_class",
        }
        class_width = 9
        feature_width = 18
        type_width = 10
        lines.extend(["", "Coefficients:", "-" * 78])
        lines.append(
            f"{'Class':<{class_width}} {'Feature':<{feature_width}} "
            f"{'Type':<{type_width}} {'Coef':>9} {'Std.Err':>9} {'z':>7} {'P>|z|':>8}"
        )
        lines.append("-" * 78)
        for row in coef_rows:
            class_label = str(row["class"])[:class_width]
            feature = str(row["feature"])[:feature_width]
            coefficient_type = coefficient_type_labels.get(
                str(row["coefficient_type"]), str(row["coefficient_type"])
            )[:type_width]
            lines.append(
                f"{class_label:<{class_width}} {feature:<{feature_width}} "
                f"{coefficient_type:<{type_width}} "
                f"{fmt(row['estimate'], width=9)} {fmt(row['std_error'], width=9)} "
                f"{fmt(row['z'], width=7, precision=3)} {fmt_p(row['p_value'])}"
            )
        lines.append("-" * 78)

        if self._smooth_results:
            lines.append(
                "Note: smooth standard errors use the penalized-fit Hessian and are "
                "naive for smoothing-parameter selection."
            )
        elif self.alpha > 0.0:
            lines.append(
                "Note: baseline-category penalties are reference-dependent; unpenalized "
                "fits are the reference-invariant path."
            )
        if self._constraint_metadata:
            lines.append(
                "Note: monotonicity constraints apply to class utilities versus the "
                "reference class, not to class probabilities."
            )
        solver_warnings = self.warnings
        if solver_warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in solver_warnings)
        return "\n".join(lines)

    def to_pmml(self, path: str | None = None, n_grid_points: int = 200) -> str:
        from rustystats.export_pmml import to_pmml

        return to_pmml(self, path=path, n_grid_points=n_grid_points)

    def to_onnx(
        self,
        path: str | None = None,
        n_grid_points: int = 200,
        mode: str = "scoring",
    ) -> bytes:
        from rustystats.export_onnx import to_onnx

        return to_onnx(self, path=path, n_grid_points=n_grid_points, mode=mode)

    def to_bytes(self) -> bytes:
        builder_state = None
        if self._builder is not None:
            builder_state = {
                "parsed_formula": self._builder._parsed_formula,
                "cat_encoding_cache": self._builder._cat_encoding_cache,
                "fitted_splines": self._builder._fitted_splines,
                "te_stats": getattr(self._builder, "_te_stats", {}),
                "fe_stats": getattr(self._builder, "_fe_stats", {}),
                "dtype": self._builder.dtype,
                "term_slots": getattr(self._builder, "_term_slots", []),
            }
        state = {
            "schema_version": _SCHEMA_VERSION,
            "result_state": {
                "params": self.params,
                "alternative_generic_coefficients": self.alternative_generic_coefficients,
                "alternative_specific_coefficients": self.alternative_specific_coefficients,
                "fitted_probabilities": self.fitted_probabilities,
                "linear_predictor": self.linear_predictor,
                "log_likelihood": self.log_likelihood,
                "deviance": self.deviance,
                "null_deviance": self.null_deviance,
                "iterations": self.iterations,
                "converged": self.converged,
                "covariance_unscaled": self._covariance(),
                "prior_weights": np.asarray(self._result.prior_weights, dtype=np.float64),
                "y_codes": np.asarray(self._result.y_codes, dtype=np.int64),
                "reference_index": self.reference_index_,
                "warnings": self.warnings,
                "solver_status": self.solver_status,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "fit_intercept": bool(getattr(self._result, "fit_intercept", True)),
                "smooth_edfs": np.asarray(
                    getattr(self._result, "smooth_edfs", np.zeros(0, dtype=np.float64)),
                    dtype=np.float64,
                ),
                "total_edf": getattr(self._result, "total_edf", None),
            },
            "response": self.response,
            "classes": self.classes_,
            "reference": self.reference_,
            "feature_names": self.feature_names,
            "builder_state": builder_state,
            "terms": self._terms_dict,
            "alternative_terms": self._alternative_terms_spec,
            "target_encoding_state": self._target_encoding_state,
            "interactions": self._interactions_spec,
            "input_transforms": self._input_transforms,
            "availability_spec": None
            if self._array_availability_requires_prediction_override
            else self._availability_spec,
            "offset_spec": None
            if self._array_offset_requires_prediction_override
            else self._offset_spec,
            "weights_spec": self._weights_spec,
            "fit_row_weights": None
            if self._fit_row_weights is None
            else np.asarray(self._fit_row_weights, dtype=np.float64),
            "array_weighted": self._array_weighted,
            "array_availability_requires_prediction_override": self._array_availability_requires_prediction_override,
            "array_offset_requires_prediction_override": self._array_offset_requires_prediction_override,
            "inference_status": self.inference_status,
            "regularization_path_info": self._regularization_path_info,
            "smooth_results": self._smooth_results,
            "total_edf": self._total_edf,
            "gcv": self._gcv,
            "smooth_profile": self._smooth_profile,
            "constraint_metadata": self._constraint_metadata,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> MultinomialModel:
        state = pickle.loads(data)
        schema_version = state.get("schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise ValidationError(
                f"Cannot load multinomial model: serialized schema_version "
                f"{schema_version!r} is not supported."
            )
        result_state = dict(state["result_state"])
        result_state.setdefault("alternative_generic_coefficients", np.zeros(0, dtype=np.float64))
        result_state.setdefault(
            "alternative_specific_coefficients",
            np.zeros((len(state["classes"]) - 1, 0), dtype=np.float64),
        )
        result_state.setdefault("smooth_edfs", np.zeros(0, dtype=np.float64))
        result_state.setdefault("total_edf", None)
        result = _DeserializedMultinomialResult(**result_state)
        builder = None
        if state["builder_state"] is not None:
            builder = _DeserializedBuilder(state["builder_state"])
        return cls(
            result=result,
            response=state.get("response"),
            classes=state["classes"],
            reference=state["reference"],
            feature_names=state["feature_names"],
            builder=builder,
            terms=state["terms"],
            alternative_terms=state.get("alternative_terms", {}),
            interactions=state["interactions"],
            input_transforms=state.get("input_transforms", []),
            compiled_input_transforms=None,
            target_encoding_state=state.get("target_encoding_state"),
            availability_spec=state.get("availability_spec"),
            offset_spec=state.get("offset_spec"),
            weights_spec=state.get("weights_spec"),
            array_availability_requires_prediction_override=state.get(
                "array_availability_requires_prediction_override", False
            ),
            array_offset_requires_prediction_override=state.get(
                "array_offset_requires_prediction_override", False
            ),
            inference_status=state.get("inference_status", "valid_standard"),
            fit_row_weights=state.get("fit_row_weights"),
            array_weighted=state.get("array_weighted", False),
            regularization_path_info=state.get("regularization_path_info"),
            smooth_results=state.get("smooth_results", []),
            total_edf=state.get("total_edf"),
            gcv=state.get("gcv"),
            smooth_profile=state.get("smooth_profile"),
            constraint_metadata=state.get("constraint_metadata", []),
        )

    def __repr__(self) -> str:
        return (
            f"<MultinomialModel: {len(self.classes_)} classes, reference={self.reference_!r}, "
            f"{self.n_params} parameters, deviance={self.deviance:.2f}>"
        )


class MultinomialDict:
    """Unfitted dict-specified multinomial model."""

    def __init__(
        self,
        *,
        response: str,
        terms: dict[str, dict[str, Any]],
        alternative_terms: dict[str, dict[str, Any]] | None,
        data: Any,
        interactions: list[dict[str, Any]] | None,
        intercept: bool,
        classes: list[str] | None,
        reference: str | None,
        availability: dict[str, str | bool | np.ndarray] | np.ndarray | None,
        weights: str | np.ndarray | None,
        class_weights: dict[str, float] | None,
        offset: dict[str, str | np.ndarray] | np.ndarray | None,
        seed: int | None,
        input_transforms: list[dict[str, Any]] | None,
    ):
        _validate_supported_terms(terms, interactions)

        self.response = response
        self.terms = copy.deepcopy(terms)
        self._raw_alternative_terms = copy.deepcopy(alternative_terms)
        self.interactions_spec = copy.deepcopy(interactions)
        self.intercept = intercept
        self.availability_spec = copy.deepcopy(availability)
        self.weights_spec = weights
        self.class_weights = copy.deepcopy(class_weights)
        self.offset_spec = copy.deepcopy(offset)
        self._seed = seed

        data_schema = None
        if hasattr(data, "collect_schema"):
            data_schema = dict(data.collect_schema())
        elif hasattr(data, "schema"):
            data_schema = dict(data.schema)
        self._input_transforms = validate_input_transforms(
            input_transforms, data_schema=data_schema
        )

        needed = _extract_needed_columns(
            self.terms,
            response=response,
            interactions=self.interactions_spec,
            input_transforms=self._input_transforms,
            weights=weights,
        )
        needed |= _target_encoding_source_columns(
            self.terms,
            self.interactions_spec,
            self._input_transforms,
        )
        needed |= _extra_needed_columns([availability, offset], self._input_transforms)
        needed |= _alternative_needed_columns(alternative_terms, self._input_transforms)
        needed |= set(input_transform_source_columns(self._input_transforms))
        data = _collect_lazyframe(data, needed)
        self._compiled_input_transforms = compile_input_transforms(
            self._input_transforms,
            assume_validated=True,
        )
        if self._compiled_input_transforms:
            data = apply_input_transforms(data, self._compiled_input_transforms)
        self._owned_transformed_data = data
        self._data_ref = weakref.ref(self._owned_transformed_data)

        if response not in data.columns:
            raise ValidationError(f"response column {response!r} is not present in data.")
        response_values = _string_labels(data[response].to_list())
        (
            self.classes_,
            self.reference_,
            self._class_to_code,
            self.y_codes,
        ) = _resolve_classes(response_values, data[response], classes, reference)
        self.reference_index_ = self._class_to_code[self.reference_]
        self.alternative_terms = _normalize_alternative_terms(
            alternative_terms, self.classes_, self.reference_
        )

        dummy_response = _unique_temp_column(data.columns, "__rustystats_multinomial_y__")
        import polars as pl

        build_data = data.with_columns(pl.Series(dummy_response, np.zeros(len(data))))
        parsed_with_target_encoding = dict_to_parsed_formula(
            response=dummy_response,
            terms=self.terms,
            interactions=self.interactions_spec,
            intercept=intercept,
        )
        shared_parsed = _parsed_without_target_encoding(parsed_with_target_encoding)
        self._builder = InteractionBuilder(build_data)
        _dummy_y, self.X, self.feature_names = self._builder.build_design_matrix_from_parsed(
            shared_parsed, seed=seed
        )
        (
            self._bound_nonneg_indices,
            self._bound_nonpos_indices,
            self._shared_bound_nonneg_indices,
            self._shared_bound_nonpos_indices,
        ) = _multinomial_bound_indices(
            self.feature_names,
            n_classes=len(self.classes_),
            reference_index=self.reference_index_,
        )
        self._constraint_metadata = _multinomial_constraint_metadata(
            self.feature_names,
            self.classes_,
            self.reference_,
            self._shared_bound_nonneg_indices,
            self._shared_bound_nonpos_indices,
        )
        (
            self.alternative_generic,
            self.alternative_specific,
            self.alternative_generic_feature_names,
            self.alternative_specific_feature_names,
        ) = _resolve_alternative_arrays(data, self.classes_, self.alternative_terms)
        self.availability = _resolve_class_matrix(
            data,
            availability,
            self.classes_,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=True,
        )
        if not np.all(self.availability[np.arange(len(self.y_codes)), self.y_codes]):
            raise ValidationError("observed response class must be available for every row.")
        self.offset = _resolve_class_matrix(
            data,
            offset,
            self.classes_,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=True,
        )
        self.weights, self._class_weighted, self._row_weights = _resolve_weights(
            data, weights, class_weights, response_values
        )
        te_specific, self._target_encoding_state = _build_multinomial_target_encoding(
            data,
            parsed_with_target_encoding.target_encoding_terms,
            classes=self.classes_,
            reference=self.reference_,
            y_codes=self.y_codes,
            availability=self.availability,
            row_weights=self._row_weights,
            seed=seed,
        )
        self.alternative_specific = _append_alternative_specific_terms(
            self.alternative_specific,
            te_specific,
        )
        if self._target_encoding_state is not None:
            self.alternative_specific_feature_names.extend(
                self._target_encoding_state.feature_names
            )
        self.n_obs = len(data)
        self.n_params = (
            self.X.shape[1] * (len(self.classes_) - 1)
            + self.alternative_generic.shape[2]
            + self.alternative_specific.shape[2] * (len(self.classes_) - 1)
        )
        self._array_weighted = weights is not None and not isinstance(weights, str)
        self._array_availability_requires_prediction_override = _spec_has_array(availability)
        self._array_offset_requires_prediction_override = _spec_has_array(offset)

    @property
    def data(self) -> Any:
        data = self._data_ref()
        if data is None:
            raise ValidationError(
                "Original DataFrame has been garbage collected. Keep a reference to the "
                "DataFrame if you need to access it after fitting."
            )
        return data

    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        regularization: str | None = None,
        cv: int | None = None,
        selection: str = "min",
        n_alphas: int = DEFAULT_N_ALPHAS,
        alphas: np.ndarray | list[float] | None = None,
        alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
        cv_seed: int | None = None,
        include_unregularized: bool = True,
        n_lambda: int = DEFAULT_N_LAMBDA,
        lambda_min: float = DEFAULT_LAMBDA_MIN,
        lambda_max: float = DEFAULT_LAMBDA_MAX,
        max_lambda_iter: int = 6,
        max_iter: int = 100,
        tol: float = 1e-8,
        standardize: bool = True,
        compute_covariance: bool = True,
        store_design_matrix: bool = False,
        verbose: bool = False,
        hessian_memory_limit_bytes: int = _DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES,
        max_dense_parameters: int = _DEFAULT_MAX_DENSE_PARAMETERS,
    ) -> MultinomialModel:
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValidationError("tol must be finite and positive.")
        path_info = None
        smooth_terms, _smooth_col_ranges = self._builder.get_smooth_terms()
        has_smooth = bool(smooth_terms)
        if has_smooth and cv is not None:
            raise ValidationError("multinomial smooth terms do not yet support cv=.")
        if cv is not None:
            path_info = _fit_multinomial_cv_path(
                self,
                regularization=regularization,
                l1_ratio=l1_ratio,
                cv=cv,
                selection=selection,
                n_alphas=n_alphas,
                alphas=alphas,
                alpha_min_ratio=alpha_min_ratio,
                cv_seed=cv_seed if cv_seed is not None else self._seed,
                include_unregularized=include_unregularized,
                max_iter=max_iter,
                tol=tol,
                standardize=standardize,
                verbose=verbose,
                hessian_memory_limit_bytes=int(hessian_memory_limit_bytes),
                max_dense_parameters=int(max_dense_parameters),
            )
            alpha = path_info.selected_alpha
            l1_ratio = path_info.selected_l1_ratio
        else:
            alpha, l1_ratio = _resolve_regularization(alpha, l1_ratio, regularization)

        if has_smooth:
            if alpha > 0.0 or l1_ratio > 0.0:
                raise ValidationError(
                    "multinomial smooth terms cannot yet be combined with ridge, lasso, "
                    "elastic_net, or alpha regularization."
                )
            (
                result,
                smooth_results,
                total_edf,
                gcv,
                smooth_profile,
            ) = _fit_multinomial_smooth_path(
                self,
                max_iter=max_iter,
                tol=tol,
                compute_covariance=compute_covariance,
                store_design_matrix=store_design_matrix,
                verbose=verbose,
                hessian_memory_limit_bytes=int(hessian_memory_limit_bytes),
                max_dense_parameters=int(max_dense_parameters),
                n_lambda=n_lambda,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                max_lambda_iter=max_lambda_iter,
            )
            self._builder.clear_caches()

            inference_notes: list[str] = ["naive_after_regularization"]
            if self._bound_nonneg_indices or self._bound_nonpos_indices:
                inference_notes.append("constrained_boundary")
            if self._class_weighted:
                inference_notes.append("naive_class_weighted")
            if not compute_covariance:
                inference_notes.append("covariance_skipped")
            elif result.cov_params_unscaled is None:
                inference_notes.append("covariance_unavailable")
            inference_status = "+".join(inference_notes)

            return MultinomialModel(
                result=result,
                response=self.response,
                classes=self.classes_,
                reference=self.reference_,
                feature_names=self.feature_names,
                builder=self._builder,
                terms=self.terms,
                alternative_terms=self.alternative_terms,
                interactions=self.interactions_spec,
                input_transforms=self._input_transforms,
                compiled_input_transforms=self._compiled_input_transforms,
                target_encoding_state=self._target_encoding_state,
                availability_spec=None
                if self._array_availability_requires_prediction_override
                else self.availability_spec,
                offset_spec=None
                if self._array_offset_requires_prediction_override
                else self.offset_spec,
                weights_spec=self.weights_spec if isinstance(self.weights_spec, str) else None,
                array_availability_requires_prediction_override=self._array_availability_requires_prediction_override,
                array_offset_requires_prediction_override=self._array_offset_requires_prediction_override,
                inference_status=inference_status,
                fit_row_weights=self._row_weights,
                array_weighted=self._array_weighted,
                regularization_path_info=None,
                smooth_results=smooth_results,
                total_edf=total_edf,
                gcv=gcv,
                smooth_profile=smooth_profile,
                constraint_metadata=self._constraint_metadata,
            )

        l1_active = alpha > 0.0 and l1_ratio > 0.0
        final_initial_result = _full_data_multinomial_warm_start_result(
            self,
            path_info,
            max_iter=max_iter,
            tol=tol,
            standardize=standardize,
            verbose=verbose,
            hessian_memory_limit_bytes=int(hessian_memory_limit_bytes),
            max_dense_parameters=int(max_dense_parameters),
        )
        if path_info is not None and path_info.cv_profile is not None:
            path_info.cv_profile["final_refit_warm_start"] = final_initial_result is not None

        result = _fit_multinomial_arrays(
            y_codes=self.y_codes,
            x=self.X,
            n_classes=len(self.classes_),
            reference_index=self.reference_index_,
            availability=self.availability,
            offset=self.offset,
            weights=self.weights,
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=self.intercept,
            standardize=standardize,
            compute_covariance=compute_covariance,
            store_design_matrix=store_design_matrix,
            verbose=verbose,
            hessian_memory_limit_bytes=int(hessian_memory_limit_bytes),
            max_dense_parameters=int(max_dense_parameters),
            alternative_generic=self.alternative_generic,
            alternative_specific=self.alternative_specific,
            initial_result=final_initial_result,
            bound_nonneg_indices=self._bound_nonneg_indices,
            bound_nonpos_indices=self._bound_nonpos_indices,
        )
        self._builder.clear_caches()

        inference_notes: list[str] = []
        if path_info is not None:
            inference_notes.append("naive_after_cv_selection")
        elif alpha > 0.0:
            inference_notes.append(
                "naive_after_selection" if l1_ratio > 0.0 else "naive_after_regularization"
            )
        if self._class_weighted:
            inference_notes.append("naive_class_weighted")
        if self._bound_nonneg_indices or self._bound_nonpos_indices:
            inference_notes.append("constrained_boundary")
        if not compute_covariance:
            inference_notes.append("covariance_skipped")
        elif l1_active or result.cov_params_unscaled is None:
            inference_notes.append("covariance_unavailable")
        inference_status = "+".join(inference_notes) if inference_notes else "valid_standard"

        return MultinomialModel(
            result=result,
            response=self.response,
            classes=self.classes_,
            reference=self.reference_,
            feature_names=self.feature_names,
            builder=self._builder,
            terms=self.terms,
            alternative_terms=self.alternative_terms,
            interactions=self.interactions_spec,
            input_transforms=self._input_transforms,
            compiled_input_transforms=self._compiled_input_transforms,
            target_encoding_state=self._target_encoding_state,
            availability_spec=None
            if self._array_availability_requires_prediction_override
            else self.availability_spec,
            offset_spec=None
            if self._array_offset_requires_prediction_override
            else self.offset_spec,
            weights_spec=self.weights_spec if isinstance(self.weights_spec, str) else None,
            array_availability_requires_prediction_override=self._array_availability_requires_prediction_override,
            array_offset_requires_prediction_override=self._array_offset_requires_prediction_override,
            inference_status=inference_status,
            fit_row_weights=self._row_weights,
            array_weighted=self._array_weighted,
            regularization_path_info=path_info,
            constraint_metadata=self._constraint_metadata,
        )


def build_multinomial_fold_design(
    model: MultinomialDict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int | None = None,
) -> MultinomialFoldDesign:
    """Build one fold's multinomial inputs with fold-local target encoding.

    Target-encoding state is fit on ``train_idx`` rows only. Training rows use
    ordered/permutation TE inside the fold, while validation rows are transformed
    through the fold-training state so validation labels never enter validation
    encodings. This is preprocessing only; multinomial CV remains disabled until
    the later regularization-path phase.
    """
    if not isinstance(model, MultinomialDict):
        raise ValidationError("build_multinomial_fold_design expects a MultinomialDict instance.")
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    if train_idx.ndim != 1 or val_idx.ndim != 1:
        raise ValidationError("train_idx and val_idx must be one-dimensional.")
    n = model.n_obs
    if np.any(train_idx < 0) or np.any(train_idx >= n):
        raise ValidationError("train_idx contains row indices outside the training data.")
    if np.any(val_idx < 0) or np.any(val_idx >= n):
        raise ValidationError("val_idx contains row indices outside the training data.")
    if np.unique(train_idx).size != train_idx.size:
        raise ValidationError("train_idx contains duplicate row indices.")
    if np.unique(val_idx).size != val_idx.size:
        raise ValidationError("val_idx contains duplicate row indices.")
    if np.intersect1d(train_idx, val_idx, assume_unique=True).size:
        raise ValidationError(
            "train_idx and val_idx must be disjoint; overlapping rows would leak validation data."
        )

    data = model.data
    data_train = data[train_idx]
    data_val = data[val_idx]
    y_train = model.y_codes[train_idx]
    y_val = model.y_codes[val_idx]
    availability_train = model.availability[train_idx, :]
    availability_val = model.availability[val_idx, :]
    offset_train = model.offset[train_idx, :]
    offset_val = model.offset[val_idx, :]
    weights_train = None if model.weights is None else model.weights[train_idx]
    weights_val = None if model.weights is None else model.weights[val_idx]
    row_weights_train = None if model._row_weights is None else model._row_weights[train_idx]

    dummy_response = _unique_temp_column(data.columns, "__rustystats_multinomial_fold_y__")
    import polars as pl

    parsed_with_target_encoding = dict_to_parsed_formula(
        response=dummy_response,
        terms=model.terms,
        interactions=model.interactions_spec,
        intercept=model.intercept,
    )
    shared_parsed = _parsed_without_target_encoding(parsed_with_target_encoding)
    shared_parsed = _fold_local_parsed_formula(shared_parsed)

    train_build_data = data_train.with_columns(pl.Series(dummy_response, np.zeros(len(data_train))))
    val_build_data = data_val.with_columns(pl.Series(dummy_response, np.zeros(len(data_val))))
    builder = InteractionBuilder(train_build_data)
    _dummy_y, x_train, feature_names = builder.build_design_matrix_from_parsed(
        shared_parsed,
        seed=seed if seed is not None else model._seed,
    )
    x_val = builder.transform_new_data(val_build_data)

    (
        alternative_generic_train,
        alternative_specific_train,
        alternative_generic_feature_names,
        alternative_specific_feature_names,
    ) = _resolve_alternative_arrays(data_train, model.classes_, model.alternative_terms)
    alternative_generic_val, alternative_specific_val, _generic_val_names, _specific_val_names = (
        _resolve_alternative_arrays(data_val, model.classes_, model.alternative_terms)
    )

    te_train, target_encoding_state = _build_multinomial_target_encoding(
        data_train,
        parsed_with_target_encoding.target_encoding_terms,
        classes=model.classes_,
        reference=model.reference_,
        y_codes=y_train,
        availability=availability_train,
        row_weights=row_weights_train,
        seed=seed if seed is not None else model._seed,
    )
    alternative_specific_train = _append_alternative_specific_terms(
        alternative_specific_train,
        te_train,
    )
    if target_encoding_state is not None:
        alternative_specific_val = _append_alternative_specific_terms(
            alternative_specific_val,
            target_encoding_state.transform(data_val),
        )
        alternative_specific_feature_names = [
            *alternative_specific_feature_names,
            *target_encoding_state.feature_names,
        ]

    return MultinomialFoldDesign(
        x_train=x_train,
        x_val=x_val,
        alternative_generic_train=alternative_generic_train,
        alternative_generic_val=alternative_generic_val,
        alternative_specific_train=alternative_specific_train,
        alternative_specific_val=alternative_specific_val,
        availability_train=availability_train,
        availability_val=availability_val,
        offset_train=offset_train,
        offset_val=offset_val,
        weights_train=weights_train,
        weights_val=weights_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=list(feature_names),
        preprocessing_state=MultinomialFoldPreprocessingState(
            builder=builder,
            parsed_formula=shared_parsed,
            target_encoding_state=target_encoding_state,
            alternative_generic_feature_names=alternative_generic_feature_names,
            alternative_specific_feature_names=alternative_specific_feature_names,
        ),
    )


def multinomial_dict(
    response: str,
    data: Any,
    *,
    terms: dict[str, dict[str, Any]] | None = None,
    shared_terms: dict[str, dict[str, Any]] | None = None,
    alternative_terms: dict[str, dict[str, Any]] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    intercept: bool = True,
    classes: list[str] | None = None,
    reference: str | None = None,
    availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
    class_weights: dict[str, float] | None = None,
    offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
    seed: int | None = None,
    input_transforms: list[dict[str, Any]] | None = None,
) -> MultinomialDict:
    """Create a native baseline-category multinomial logit model from a dict spec."""
    if terms is not None and shared_terms is not None:
        raise ValidationError("Pass only one of terms= or shared_terms=.")
    resolved_terms = terms if terms is not None else shared_terms
    if resolved_terms is None:
        resolved_terms = {}
    if not isinstance(resolved_terms, dict):
        raise ValidationError("terms/shared_terms must be a dict.")
    return MultinomialDict(
        response=response,
        terms=resolved_terms,
        alternative_terms=alternative_terms,
        data=data,
        interactions=interactions,
        intercept=intercept,
        classes=classes,
        reference=reference,
        availability=availability,
        weights=weights,
        class_weights=class_weights,
        offset=offset,
        seed=seed,
        input_transforms=input_transforms,
    )


__all__ = [
    "MultinomialFoldDesign",
    "MultinomialFoldPreprocessingState",
    "MultinomialDatasetDiagnostics",
    "MultinomialDiagnostics",
    "MultinomialDict",
    "MultinomialModel",
    "MultinomialScenario",
    "build_multinomial_fold_design",
    "multinomial_alpha_max",
    "multinomial_dict",
]
