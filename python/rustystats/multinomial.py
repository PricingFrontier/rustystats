"""Native baseline-category multinomial logit API."""

from __future__ import annotations

import copy
import json
import math
import pickle
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from rustystats._rustystats import fit_multinomial_py as _fit_multinomial_rust
from rustystats.exceptions import PredictionError, ValidationError
from rustystats.formula import (
    _collect_lazyframe,
    _compute_predict_chunk_size,
    _extract_needed_columns,
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
from rustystats.interactions import InteractionBuilder

_DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_DENSE_PARAMETERS = 5000
_MIN_WEIGHTED_STD = 1e-12
_SCHEMA_VERSION = 1


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


def _validate_supported_term_spec(var_name: str, spec: dict[str, Any], *, context: str) -> None:
    term_type = spec.get("type", "linear")
    if term_type == "target_encoding":
        raise ValidationError(
            "target_encoding is not yet supported for multinomial_dict. Use categorical or "
            "frequency_encoding terms, or precompute multiclass encodings outside the model."
        )
    if term_type in {"ms", "s"}:
        raise ValidationError(
            f"{term_type} smooth/monotonic spline terms are not yet supported for "
            "multinomial_dict. Use fixed-degree bs/ns terms without monotonicity."
        )
    if spec.get("monotonicity") is not None:
        raise ValidationError(
            f"monotonicity constraints are not yet supported for multinomial_dict "
            f"({context} term {var_name!r})."
        )
    if term_type in {"bs", "ns"}:
        if spec.get("k") is not None:
            raise ValidationError(
                f"automatic smooth penalties are not yet supported for multinomial_dict "
                f"({context} term {var_name!r}); use df= or knots= for a fixed basis."
            )
        if spec.get("df") is None and spec.get("knots") is None:
            raise ValidationError(
                f"multinomial_dict requires fixed-degree {term_type} terms. "
                f"Pass df= or knots= for {context} term {var_name!r}."
            )
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
    }
    for interaction in interactions or []:
        if interaction.get("target_encoding"):
            raise ValidationError(
                "target_encoding interactions are not yet supported for multinomial_dict."
            )
        for var_name, spec in interaction.items():
            if var_name in reserved:
                continue
            if not isinstance(spec, dict):
                raise ValidationError(f"interaction spec for {var_name!r} must be a dict.")
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
) -> tuple[np.ndarray | None, bool]:
    row_weights = None
    if weights is not None:
        if isinstance(weights, str):
            row_weights = data[weights].to_numpy().astype(np.float64)
        else:
            row_weights = _as_float_array(weights, name="weights", length=len(data))
        if np.any(~np.isfinite(row_weights)) or np.any(row_weights < 0.0):
            raise ValidationError("weights must be finite and non-negative.")

    if class_weights is None:
        return row_weights, False

    resolved = {str(key): float(value) for key, value in class_weights.items()}
    bad = [key for key, value in resolved.items() if not np.isfinite(value) or value < 0.0]
    if bad:
        raise ValidationError(f"class_weights must be finite and non-negative; bad keys: {bad}.")

    class_multiplier = np.asarray([resolved.get(label, 1.0) for label in response_values])
    effective = class_multiplier if row_weights is None else row_weights * class_multiplier
    return effective.astype(np.float64, copy=False), True


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
        availability_spec: Any,
        offset_spec: Any,
        weights_spec: str | None,
        array_availability_requires_prediction_override: bool,
        array_offset_requires_prediction_override: bool,
        inference_status: str,
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
        self.inference_status = inference_status

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
        if self.alpha > 0.0 or "class_weighted" in self.inference_status:
            return None
        return -2.0 * self.log_likelihood + 2.0 * self.n_params

    def bic(self) -> float | None:
        if self.alpha > 0.0 or "class_weighted" in self.inference_status:
            return None
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

    def _diagnostic_weight_array(self, data: Any) -> np.ndarray:
        if self._weights_spec is None:
            return np.ones(len(data), dtype=np.float64)
        if self._weights_spec not in data.columns:
            raise PredictionError(f"weights column {self._weights_spec!r} is not present in data.")
        weights = _as_float_array(
            data[self._weights_spec].to_numpy(), name="weights", length=len(data)
        )
        if np.any(weights < 0.0):
            raise ValidationError("weights must be non-negative.")
        return weights

    def _collect_diagnostic_data(
        self,
        data: Any,
        *,
        categorical_factors: list[str],
        continuous_factors: list[str],
    ) -> Any:
        if self.response is None:
            raise PredictionError(
                "Cannot compute diagnostics on supplied data: this model has no stored response "
                "column metadata."
            )

        requested = {self.response, *categorical_factors, *continuous_factors}
        if self._weights_spec is not None:
            requested.add(self._weights_spec)
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
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
        diagnostic_data = self._collect_diagnostic_data(
            data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
        )
        probabilities = self.predict_proba(data, return_format="numpy")
        if probabilities.shape[0] != len(diagnostic_data):
            raise PredictionError("Diagnostic data and prediction row counts do not match.")
        y_codes = self._class_codes_from_values(
            diagnostic_data[self.response].to_list(), name=self.response or "response"
        )
        weights = self._diagnostic_weight_array(diagnostic_data)
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
    ) -> MultinomialDiagnostics:
        categorical_factors = list(categorical_factors or [])
        continuous_factors = list(continuous_factors or [])
        factor_diagnostics: list[dict[str, Any]] = []

        using_fitted_training = train_data is None
        if using_fitted_training:
            if categorical_factors or continuous_factors:
                raise ValidationError("factor diagnostics require train_data.")
            train = self._dataset_diagnostics_from_arrays(
                name="train",
                probabilities=self.fitted_probabilities,
                y_codes=np.asarray(self._result.y_codes, dtype=np.int64),
                weights=np.asarray(self._result.prior_weights, dtype=np.float64),
            )
        else:
            train_frame, train_probabilities, train_y, train_weights = (
                self._diagnostic_arrays_from_data(
                    train_data,
                    categorical_factors=categorical_factors,
                    continuous_factors=continuous_factors,
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
        pseudo_r2 = None
        if self.null_deviance > 0.0:
            pseudo_r2 = 1.0 - train.deviance / self.null_deviance

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
            "class-mix calibration; explicit multinomial calibration objects are planned "
            "for a later phase."
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
            offset_chunk = self._resolve_prediction_offset(chunk, offset)
            logits_chunk = offset_chunk
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
        return_format: str = "numpy",
    ) -> np.ndarray:
        data = self._prepare_prediction_data(new_data, availability, offset)
        logits = self.decision_function(data, availability=availability, offset=offset)
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
        return_format: str = "numpy",
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            new_data, availability=availability, offset=offset, return_format="numpy"
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
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            new_data, availability=availability, offset=offset, return_format="numpy"
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
    ) -> Any:
        if k <= 0 or k > len(self.classes_):
            raise ValidationError(f"k must be in [1, {len(self.classes_)}].")
        probabilities = self.predict_proba(
            new_data, availability=availability, offset=offset, return_format="numpy"
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
        return_format: str = "dict",
    ) -> dict[str, float]:
        data = self._prepare_prediction_data(new_data, availability, offset)
        probabilities = self.predict_proba(
            data, availability=availability, offset=offset, return_format="numpy"
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
            data, availability=availability, offset=offset, return_format="numpy"
        )
        scenario_probabilities = self.predict_proba(
            scenario_data, availability=availability, offset=offset, return_format="numpy"
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
        lines = [
            "=" * 78,
            "Multinomial Logit Results".center(78),
            "=" * 78,
            f"{'Classes:':<20} {', '.join(self.classes_)}",
            f"{'Reference:':<20} {self.reference_}",
            f"{'No. Observations:':<20} {self.nobs:>10}",
            f"{'Parameters:':<20} {self.n_params:>10}",
            f"{'Method:':<20} {'Newton' if self.alpha == 0.0 else 'Newton + Ridge'}",
            f"{'Iterations:':<20} {self.iterations:>10}",
            f"{'Converged:':<20} {self.converged!s:>10}",
            f"{'Solver status:':<20} {self.solver_status}",
            "",
            f"{'Log-Likelihood:':<20} {self.log_likelihood:>15.4f}",
            f"{'Deviance:':<20} {self.deviance:>15.4f}",
            f"{'Null Deviance:':<20} {self.null_deviance:>15.4f}",
            f"{'AIC:':<20} {'NA' if aic is None else f'{aic:.4f}':>15}",
            f"{'BIC:':<20} {'NA' if bic is None else f'{bic:.4f}':>15}",
            f"{'Log Loss:':<20} {diagnostics.log_loss:>15.4f}",
            f"{'Accuracy:':<20} {diagnostics.accuracy:>15.4f}",
            f"{'Top-2 Accuracy:':<20} {diagnostics.top_2_accuracy:>15.4f}",
            f"{'Inference:':<20} {self.inference_status}",
            "=" * 78,
        ]
        if self.alpha > 0.0:
            lines.append(
                "Note: baseline ridge is reference-dependent; unpenalized fits are the "
                "reference-invariant path."
            )
        solver_warnings = self.warnings
        if solver_warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in solver_warnings)
        return "\n".join(lines)

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
            },
            "response": self.response,
            "classes": self.classes_,
            "reference": self.reference_,
            "feature_names": self.feature_names,
            "builder_state": builder_state,
            "terms": self._terms_dict,
            "alternative_terms": self._alternative_terms_spec,
            "interactions": self._interactions_spec,
            "input_transforms": self._input_transforms,
            "availability_spec": None
            if self._array_availability_requires_prediction_override
            else self._availability_spec,
            "offset_spec": None
            if self._array_offset_requires_prediction_override
            else self._offset_spec,
            "weights_spec": self._weights_spec,
            "array_availability_requires_prediction_override": self._array_availability_requires_prediction_override,
            "array_offset_requires_prediction_override": self._array_offset_requires_prediction_override,
            "inference_status": self.inference_status,
        }
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> MultinomialModel:
        state = pickle.loads(data)
        if state.get("schema_version") != _SCHEMA_VERSION:
            raise ValidationError(
                f"Cannot load multinomial model: serialized schema_version "
                f"{state.get('schema_version')!r} is not supported."
            )
        result_state = dict(state["result_state"])
        result_state.setdefault("alternative_generic_coefficients", np.zeros(0, dtype=np.float64))
        result_state.setdefault(
            "alternative_specific_coefficients",
            np.zeros((len(state["classes"]) - 1, 0), dtype=np.float64),
        )
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

        self._input_transforms = validate_input_transforms(
            input_transforms, data_schema=None if not hasattr(data, "schema") else dict(data.schema)
        )

        needed = _extract_needed_columns(
            self.terms,
            response=response,
            interactions=self.interactions_spec,
            input_transforms=self._input_transforms,
            weights=weights,
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
        self._owned_transformed_data = data if self._compiled_input_transforms else None
        self._data_ref = weakref.ref(data)

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
        self.alternative_terms = _normalize_alternative_terms(alternative_terms, self.classes_)

        dummy_response = _unique_temp_column(data.columns, "__rustystats_multinomial_y__")
        import polars as pl

        build_data = data.with_columns(pl.Series(dummy_response, np.zeros(len(data))))
        parsed = dict_to_parsed_formula(
            response=dummy_response,
            terms=self.terms,
            interactions=self.interactions_spec,
            intercept=intercept,
        )
        self._builder = InteractionBuilder(build_data)
        _dummy_y, self.X, self.feature_names = self._builder.build_design_matrix_from_parsed(
            parsed, seed=seed
        )
        (
            self.alternative_generic,
            self.alternative_specific,
            self.alternative_generic_feature_names,
            self.alternative_specific_feature_names,
        ) = _resolve_alternative_arrays(data, self.classes_, self.alternative_terms)
        self.n_obs = len(data)
        self.n_params = (
            self.X.shape[1] * (len(self.classes_) - 1)
            + self.alternative_generic.shape[2]
            + self.alternative_specific.shape[2] * (len(self.classes_) - 1)
        )

        self.availability = _resolve_class_matrix(
            data,
            availability,
            self.classes_,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=True,
        )
        self.offset = _resolve_class_matrix(
            data,
            offset,
            self.classes_,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=True,
        )
        self.weights, self._class_weighted = _resolve_weights(
            data, weights, class_weights, response_values
        )
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
        max_iter: int = 100,
        tol: float = 1e-8,
        standardize: bool = True,
        compute_covariance: bool = True,
        store_design_matrix: bool = False,
        verbose: bool = False,
        hessian_memory_limit_bytes: int = _DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES,
        max_dense_parameters: int = _DEFAULT_MAX_DENSE_PARAMETERS,
    ) -> MultinomialModel:
        del selection
        if cv is not None:
            raise ValidationError("cross-validation is not yet supported for multinomial_dict.")
        if regularization not in {None, "ridge"}:
            raise ValidationError("multinomial_dict supports only regularization='ridge'.")
        if l1_ratio != 0.0:
            raise ValidationError("multinomial_dict supports ridge only; use l1_ratio=0.0.")
        if alpha < 0.0 or not np.isfinite(alpha):
            raise ValidationError("alpha must be finite and non-negative.")
        if tol <= 0.0 or not np.isfinite(tol):
            raise ValidationError("tol must be finite and positive.")

        center = scale = None
        (
            alternative_generic_center,
            alternative_generic_scale,
            alternative_specific_center,
            alternative_specific_scale,
        ) = (None, None, None, None)
        if alpha > 0.0 and standardize:
            from rustystats.regularization_path import (
                compute_standardization,
                solver_standardization,
            )

            center, scale = compute_standardization(
                self.X, self.weights, fit_intercept=self.intercept
            )
            center, scale = solver_standardization(center, scale, fit_intercept=self.intercept)
            (
                alternative_generic_center,
                alternative_generic_scale,
                alternative_specific_center,
                alternative_specific_scale,
            ) = _alternative_standardization(
                self.alternative_generic,
                self.alternative_specific,
                self.availability,
                self.weights,
                self.reference_index_,
            )

        result = _fit_multinomial_rust(
            self.y_codes,
            np.ascontiguousarray(self.X, dtype=np.float64),
            len(self.classes_),
            self.reference_index_,
            self.availability,
            self.offset,
            self.weights,
            alpha,
            l1_ratio,
            max_iter,
            tol,
            self.intercept,
            center,
            scale,
            not compute_covariance,
            int(hessian_memory_limit_bytes),
            int(max_dense_parameters),
            store_design_matrix,
            verbose,
            self.alternative_generic,
            self.alternative_specific,
            alternative_generic_center,
            alternative_generic_scale,
            alternative_specific_center,
            alternative_specific_scale,
        )
        self._builder.clear_caches()

        inference_notes: list[str] = []
        if alpha > 0.0:
            inference_notes.append("naive_after_regularization")
        if self._class_weighted:
            inference_notes.append("naive_class_weighted")
        if not compute_covariance:
            inference_notes.append("covariance_skipped")
        elif result.cov_params_unscaled is None:
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
    "MultinomialDatasetDiagnostics",
    "MultinomialDiagnostics",
    "MultinomialDict",
    "MultinomialModel",
    "MultinomialScenario",
    "multinomial_dict",
]
