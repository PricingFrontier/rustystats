"""Out-of-core fitting of a dict-specified GLM (``specs/out-of-core-fit.md`` §4–§8).

``rs.glm_dict(..., data=BatchSource(...))`` returns an :class:`OutOfCoreGLMDict`.
Construction checks the model against the source's schema and reads nothing.
``fit(out_of_core=OutOfCoreOptions(...))`` then reads the source once per pass:

1. **Validate** every option and the model before any I/O.
2. **Learn** the frozen design (§6): the freeze pass (category keys, frequency counts,
   spline ranges, the response mean, the y/weight/offset checks and the source's
   fingerprints), then the knot passes, then target encoding's te_stats and te_sort
   passes, whose sort runs spill under ``scratch_dir`` (§6.6, §10).
3. **Moments** (§6.7): two passes over the encoded rows when the penalty is
   standardised.
4. **Solve** (§7): Rust drives the passes; this module serves each one, encoding each
   slice of each batch into one reused block of ``encode_rows`` × p.

Every pass cuts each source batch into slices before it touches the rows, and
fingerprints each slice of the columns it reads, comparing them with the freeze pass
when the pass ends (§9), so a source that changes between passes is refused. Each
phase's memory is planned and admitted before it starts (§11). Exposure, offset,
weights and a complement may be columns of the source or arrays the caller holds; an
array is read a slice at a time by global row. The result holds no per-row arrays
(§4.5).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from rustystats._rustystats import (
    ColumnFingerprint,
    RowSum,
    compute_standardization_batched_py,
    fit_glm_batched_py,
    fit_smooth_glm_batched_py,
    fold_alpha_max_batched_py,
    fold_path_batched_py,
    sparse_rows_py,
)
from rustystats.category_keys import (
    ENTRY_OVERHEAD_BYTES,
    CategoryKeyLearner,
    LearnedStateBudget,
    check_key_dtype,
)
from rustystats.category_keys import observed_category_labels as _column_labels
from rustystats.constants import (
    DEFAULT_ALPHA_MIN_RATIO,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_N_ALPHAS,
    DEFAULT_NEGBINOMIAL_THETA,
    DEFAULT_TOLERANCE,
)
from rustystats.exceptions import (
    BatchSourceChangedError,
    MemoryPlanExceededError,
    OutOfCoreOptionError,
    SpillError,
    ValidationError,
)
from rustystats.frozen_design import (
    FoldRows,
    FrozenDesign,
    FrozenDesignUnsupported,
    TargetInputs,
    _spline_terms,
    _target_variables,
    _training_builder,
    check_key_dtypes,
    draw_target_encoding_seed,
    learn_frozen_designs,
    learn_target_values,
)
from rustystats.input_transforms import (
    apply_input_transforms,
    compile_input_transforms,
    input_transform_source_columns,
    validate_input_transforms,
)
from rustystats.out_of_core import (
    DEFAULT_ENCODE_BLOCK_BYTES,
    DEFAULT_LEARN_SLICE_BYTES,
    E_LEARN,
    GRAM_BLOCK_ROWS,
    BatchProgress,
    BatchSource,
    OutOfCoreOptions,
    _PassAborted,
    fingerprint_update,
    slice_rows,
)
from rustystats.solver_trace import solver_steps
from rustystats.validation import (
    coerce_to_float64,
    validate_offset,
    validate_response,
    validate_weights,
)

if TYPE_CHECKING:
    from rustystats.formula import GLMModel

# ── Memory plan constants (§11) ─────────────────────────────────────────────

#: Bytes per row of the per-row work vectors the solver keeps for a slice: y, prior
#: weights, offset, η and μ per predictor state, W and z.
_WORK_BYTES_PER_ROW = 8 * (6 + 4)
#: Per encoded row and column: the block's finiteness mask (one byte per value).
_FINITE_MASK_BYTES = 1
#: The fixed allowance of the memory plan.
_ALLOCATOR_ALLOWANCE = 64 * 1024 * 1024
#: A target-encoding sort record and its sort scratch (§6.6).
_TE_RECORD_PLAN_BYTES = 2 * 40
#: Target-encoding merges: fan-in 64, each input and the output with a 1 MiB buffer.
_TE_MERGE_BYTES = (64 + 1) * 1024 * 1024
#: A learning slice's scratch per row beyond its payload (§4.3): the row-size window,
#: a distinct key's per-entry overhead in ``unique``, and the slice's float64 reads
#: of y, weights, offset and exposure with their checks, the fingerprints' arrays and
#: the fold masks.
_LEARN_ROW_BYTES = 8 + ENTRY_OVERHEAD_BYTES + 7 * 8
#: Copies of a learning slice's payload alive at once: a fold design's training rows
#: (``filter``), and a column's non-null values with their distinct keys while they
#: are merged. The fingerprint (4,096 rows at a time: one copy of String/Binary bytes,
#: at most two of categorical, enum and other labels) is done before the slice is learned.
_LEARN_PAYLOAD_COPIES = 3
#: Copies of an encode slice's variable-width payload alive at once: keys cast to
#: String with their codes, or the fingerprint's (4,096 rows at a time: one copy of
#: String/Binary bytes, at most two of categorical, enum and other labels).
_ENCODE_PAYLOAD_COPIES = 2
#: Copies of a row larger than a learning slice alive while it is processed: the
#: fingerprint's (one copy of String/Binary bytes, at most two of categorical, enum and
#: other labels), then a fold's filtered rows, a column's non-null values, its distinct
#: keys and a frequency term's grouped keys (§4.3).
_OVERSIZED_ROW_COPIES = 4
#: A slice's share of ``max_bytes`` when the learning slice is sized from it.
_LEARN_SLICE_SHARE = 16
#: Stack of each lockstep cross-validation fold's driver thread (``solvers/cv.rs``).
_FOLD_STACK_BYTES = 2 * 1024 * 1024
#: Length-p vectors a driver keeps (coefficients, the retained provenances' β, the
#: best iterate, steps and their projections).
_DRIVER_VECTORS = 16
#: Gram outputs one solver pass holds: a candidate's speculative Gram, or the
#: reconstruct pass's Gram (whose mixed right-hand side is planned beside it).
_GRAMS_IN_FLIGHT = 1
#: Shifts per thread in one intercept-refresh score tree (§7.6).
_SCORE_TREE_SHIFTS = 255
#: Rows per block read back in a θ pass beyond the encoded row: y, weights and μ from
#: the spill, and η and μ written to it (§7.10).
_THETA_ROW_BYTES = 5 * 8
#: Links the bindings accept (``families_py::link_from_name``).
_LINKS = ("identity", "log", "logit")
#: Dtypes whose values have a variable-width payload.
_VARIABLE_WIDTH = (pl.String, pl.Binary, pl.Categorical, pl.Enum)


@dataclass(frozen=True)
class MemoryPlan:
    """An out-of-core fit's planned memory (§11), reported on its result.

    ``phases`` maps each phase to its planned peak in bytes, beyond
    ``persistent_bytes``; ``max_bytes`` is the limit they were admitted against and
    ``threads`` the accumulation threads they were planned for.
    """

    phases: dict[str, int]
    persistent_bytes: int
    spill_bytes: int
    encode_rows: int
    max_bytes: int | None
    threads: int = 1


@dataclass
class _FreezeFacts:
    """What the freeze pass reads besides the design: the response mean and the
    whole-column checks the per-slice validation cannot make."""

    response_sum: Any = field(default_factory=RowSum)
    first_response: float | None = None
    response_varies: bool = False
    weight_sum: float = 0.0
    zero_weights: int = 0
    rows: int = 0


class _RowInput:
    """Exposure, offset, weights or a complement: a source column, or a caller's array
    read a slice at a time by global row (§6.8), never copied whole."""

    def __init__(self, name: str, spec: Any) -> None:
        self.name = name
        self.column = spec if isinstance(spec, str) else None
        self.array = None if isinstance(spec, str) else spec
        self.length = 0
        #: The key an array's fingerprint is kept under. Keys are tagged, so no source
        #: column name can share an array's fingerprint.
        self.key: tuple[str, str] = ("array", name)
        #: The current slice's values: (first row, rows, values), read once per slice
        #: so what is fingerprinted is what the pass uses.
        self._slice: tuple[int, int, np.ndarray] | None = None
        if self.array is not None:
            if isinstance(spec, np.ndarray) and spec.ndim != 1:
                raise ValidationError(
                    f"{name} must be a one-dimensional array with one value per row; got "
                    f"shape {spec.shape}."
                )
            try:
                self.length = len(spec)
            except TypeError as exc:
                raise ValidationError(
                    f"{name}= takes a column name or an array with one value per row, got "
                    f"{type(spec).__name__}."
                ) from exc

    def read(self, rows: pl.DataFrame, first_row: int) -> np.ndarray:
        """The values of global rows ``first_row .. first_row + len(rows)`` as float64."""
        if self.column is not None:
            return _float64(rows, self.column)
        # Every pass reads an array through the slice `_Run.slices` prepared for it.
        held = self._slice
        assert held is not None and held[:2] == (first_row, rows.height)
        return held[2]

    def prepare(self, first_row: int, n: int) -> np.ndarray:
        """Read an array's values for a pass's next slice and hold them for its
        readers (the slice's fingerprint, then the pass)."""
        values = self._read_array(first_row, n)
        self._slice = (first_row, n, values)
        return values

    def _read_array(self, first_row: int, n: int) -> np.ndarray:
        stop = first_row + n
        if stop > self.length:
            raise ValidationError(
                f"{self.name} array length {self.length} does not match the data: the source "
                f"has at least {stop} rows."
            )
        values = np.asarray(self.array[first_row:stop], dtype=np.float64)
        if values.ndim != 1:
            raise ValidationError(
                f"{self.name} must be a one-dimensional array with one value per row."
            )
        return values

    def check_rows(self, n_rows: int) -> None:
        """An array must hold exactly one value per source row (checked when the
        freeze pass has counted them)."""
        if self.array is not None and self.length != n_rows:
            raise ValidationError(
                f"{self.name} array length {self.length} does not match data length {n_rows}."
            )


class OutOfCoreGLMDict:
    """A dict-specified GLM whose rows a :class:`BatchSource` serves.

    Construction validates the model against the source's schema and reads nothing;
    :meth:`fit` learns the design and fits it in passes over the source.
    """

    def __init__(
        self,
        response: str,
        terms: dict[str, dict[str, Any]],
        data: BatchSource,
        interactions: list[dict[str, Any]] | None = None,
        intercept: bool = True,
        family: str = "gaussian",
        link: str | None = None,
        var_power: float = 1.5,
        theta: float | str | None = None,
        exposure: str | np.ndarray | None = None,
        offset: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
        seed: int | None = None,
        complement: Any = None,
        input_transforms: list[dict[str, Any]] | None = None,
        allow_extended_tweedie: bool = False,
    ) -> None:
        from rustystats.formula import (
            FormulaGLMDict,
            GLMModel,
            _extract_needed_columns,
            _GLMBase,
            _resolve_family_spec,
            dict_to_parsed_formula,
        )

        self.response = response
        self.terms = terms
        self.interactions_spec = interactions
        self.intercept = intercept
        self.source = data
        self.family, self.var_power, self.theta = _resolve_family_spec(family, var_power, theta)
        self.link = link
        self.allow_extended_tweedie = bool(allow_extended_tweedie)
        self._exposure_spec = exposure
        self._offset_spec = offset
        self._weights_spec = weights
        self._seed = seed
        # As in memory: a column or a model is kept for prediction; an array only
        # shifts the fit.
        self._complement = complement
        self._complement_spec = complement if isinstance(complement, str | GLMModel) else None
        self._complement_values = None
        self._exposure = None if exposure is None else _RowInput("exposure", exposure)
        self._offset = None if offset is None else _RowInput("offset", offset)
        self._weights = None if weights is None else _RowInput("weights", weights)
        self._complement_input = (
            None
            if complement is None or isinstance(complement, GLMModel)
            else _RowInput("complement", complement)
        )
        #: The caller's arrays among them: fingerprinted by global row in every pass,
        #: like the source's columns (§9).
        self._array_inputs = [
            row_input
            for row_input in (self._exposure, self._offset, self._weights, self._complement_input)
            if row_input is not None and row_input.array is not None
        ]
        self._input_transforms = validate_input_transforms(
            input_transforms, data_schema=dict(data.schema)
        )
        self._compiled_input_transforms = compile_input_transforms(
            self._input_transforms, assume_validated=True
        )
        self.formula = FormulaGLMDict._build_formula_string(self)
        self._parsed = dict_to_parsed_formula(
            response=response, terms=terms, interactions=interactions, intercept=intercept
        )
        if exposure is not None and not _GLMBase._uses_log_link(self):
            raise ValidationError(
                "`exposure=` is only supported for log-link rate models "
                f"(got family={self.family!r}, link={self.link!r}). Use "
                "`offset=` for a link-scale adjustment on other links."
            )
        needed = _extract_needed_columns(
            terms,
            response=response,
            interactions=interactions,
            input_transforms=self._input_transforms,
            offset=offset,
            weights=weights,
            exposure=exposure,
            complement=complement,
        )
        missing = sorted(column for column in needed if column not in data.schema)
        if missing:
            raise ValidationError(
                f"The source has no column(s) {missing}; its schema has {sorted(data.schema)}."
            )
        self._source_columns = sorted(needed)
        #: Whether the fit estimates θ, which spills μ (set by :meth:`fit`).
        self._needs_mu_spill = False
        self._check_key_dtypes()
        # Learned by fit.
        self._builder = None
        self.feature_names: list[str] | None = None

    def _check_key_dtypes(self) -> None:
        """Categorical, frequency- and target-encoded columns need a scalar dtype with a
        training key (§6.3, DEC-5), known from the schema: refused as in memory."""
        check_key_dtypes(self._parsed, self.source.schema)

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int | None = None,
        tol: float = DEFAULT_TOLERANCE,
        cv: int | None = None,
        selection: str = "min",
        regularization: str | None = None,
        n_alphas: int = DEFAULT_N_ALPHAS,
        alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
        cv_seed: int | None = None,
        cv_max_iter: int | None = None,
        cv_tol: float | None = None,
        include_unregularized: bool = True,
        standardize: bool = True,
        verbose: bool = False,
        store_design_matrix: bool = False,
        compute_covariance: bool = True,
        robust: Sequence[str] = (),
        out_of_core: OutOfCoreOptions | None = None,
    ) -> GLMModel:
        """Fit the model in passes over the source (``FormulaGLMDict.fit``'s options).

        ``out_of_core`` sets the memory limits and hooks (:class:`OutOfCoreOptions`).
        """
        from rustystats.formula import _resolve_robust_types

        options = out_of_core if out_of_core is not None else OutOfCoreOptions()
        if not isinstance(options, OutOfCoreOptions):
            raise OutOfCoreOptionError(
                f"out_of_core= takes OutOfCoreOptions, got {type(options).__name__}."
            )
        # 1. Validate everything before reading the source.
        if store_design_matrix:
            raise OutOfCoreOptionError(
                "store_design_matrix=True keeps the n × p design, which an out-of-core fit "
                "never holds. List the robust covariances you need in robust= instead."
            )
        robust_types = _resolve_robust_types(robust, compute_covariance)
        if regularization is not None and cv is None:
            cv = 5
        if cv is not None and regularization is None:
            raise ValidationError(
                "When cv is specified, 'regularization' must be set to 'ridge', 'lasso', or "
                "'elastic_net'"
            )
        if regularization is not None and regularization not in ("ridge", "lasso", "elastic_net"):
            raise ValidationError(f"Unknown regularization type: {regularization}")
        self._check_options(
            alpha=alpha,
            l1_ratio=l1_ratio,
            cv=cv,
            selection=selection,
            max_iter=max_iter,
            cv_max_iter=cv_max_iter,
        )
        nb_estimate, nb_theta = self._resolve_theta(alpha, cv, regularization, options)
        self._needs_mu_spill = nb_estimate
        if alpha < 0:
            raise ValidationError(f"alpha must be >= 0, got {alpha}.")
        if self._parsed.target_encoding_terms and options.scratch_dir is None:
            raise OutOfCoreOptionError(
                "Target encoding keeps its training values in spill files; set "
                "OutOfCoreOptions.scratch_dir."
            )
        if self._parsed.target_encoding_terms or nb_estimate:
            scratch_dir = os.fspath(options.scratch_dir)
            if not os.path.isdir(scratch_dir):
                raise SpillError(
                    f"Cannot create spill files in {scratch_dir!r}: it is not a directory."
                )

        # 2. The passes. Spill files are removed on every exit path (§4.3); an
        # exception raised while serving a pass reaches the caller unchanged (§7.1).
        run = _Run(self, options)
        try:
            result = self._fit_passes(
                run,
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                cv=cv,
                selection=selection,
                regularization=regularization,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
                cv_seed=cv_seed,
                cv_max_iter=cv_max_iter,
                cv_tol=cv_tol,
                include_unregularized=include_unregularized,
                standardize=standardize,
                verbose=verbose,
                compute_covariance=compute_covariance,
                robust_types=robust_types,
                nb_estimate=nb_estimate,
                nb_theta=nb_theta,
            )
        except _PassAborted as aborted:
            run.close(quiet=True)
            raise aborted.original from None
        except BaseException:
            run.close(quiet=True)
            raise
        run.close()
        return result

    def _check_options(
        self,
        *,
        alpha: float,
        l1_ratio: float,
        cv: int | None,
        selection: str,
        max_iter: int | None,
        cv_max_iter: int | None,
    ) -> None:
        """Refusals the schema and options decide, made before the source is opened
        (§4.3). Each is a model the in-memory fit refuses too, at the latest once it
        has fitted."""
        from rustystats.formula import _check_cv_folds

        _check_cv_folds(cv)
        if cv is not None:
            if selection not in ("min", "1se"):
                raise ValidationError(f"Unknown selection method: {selection}")
        for name, value in (("max_iter", max_iter), ("cv_max_iter", cv_max_iter)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0
            ):
                raise ValidationError(f"{name} must be a non-negative integer, got {value!r}.")
        if self.link is not None and str(self.link).lower() not in _LINKS:
            raise ValidationError(f"Unknown link '{self.link}'. Use 'identity', 'log', or 'logit'.")
        # The family's checks that need no data (an unknown family, a Tweedie power
        # outside the supported regimes) on one valid response value.
        validate_response(
            np.ones(1),
            self.family,
            require_variation=False,
            var_power=self.var_power,
            allow_extended_tweedie=self.allow_extended_tweedie,
        )
        # Solver contradictions the formula decides: a main-effect smooth always has
        # columns, and pos()/neg() (monotone linear terms) and fixed-df monotone or
        # ms() splines always constrain theirs (`_get_constraint_indices`). The same
        # refusals are made from the learned design for terms whose columns depend on
        # the data.
        parsed = self._parsed
        smooth = any(getattr(spline, "_is_smooth", False) for spline in parsed.spline_terms)
        constrained = bool(parsed.constraint_terms) or any(
            not getattr(spline, "_is_smooth", False)
            and (spline.spline_type == "ms" or bool(spline.monotonicity))
            for spline in parsed.spline_terms
        )
        if smooth and alpha > 0.0 and l1_ratio > 0.0:
            raise ValidationError(_SMOOTH_L1)
        if smooth and cv is not None:
            raise ValidationError(_SMOOTH_CV)
        if isinstance(self.theta, str) and self.theta == "estimate" and (smooth or constrained):
            raise ValidationError(
                _theta_estimate_refusal("smooth" if smooth else "sign-constrained")
            )

    def _fit_passes(
        self,
        run: _Run,
        *,
        alpha: float,
        l1_ratio: float,
        max_iter: int | None,
        tol: float,
        cv: int | None,
        selection: str,
        regularization: str | None,
        n_alphas: int,
        alpha_min_ratio: float,
        cv_seed: int | None,
        cv_max_iter: int | None,
        cv_tol: float | None,
        include_unregularized: bool,
        standardize: bool,
        verbose: bool,
        compute_covariance: bool,
        robust_types: list[str] | None,
        nb_estimate: bool,
        nb_theta: float | None,
    ) -> GLMModel:
        """:meth:`fit` after its validation: learning, then the solve."""
        from rustystats.constants import DEFAULT_CV_SEED

        # Cross-validation with target encoding learns every fold's design from its
        # training rows in the same passes as the full data's (§7.9).
        te_cv = cv is not None and bool(self._parsed.target_encoding_terms)
        cv_seed = cv_seed if cv_seed is not None else self._seed
        folds = []
        if te_cv:
            assert cv is not None
            fold_seed = cv_seed if cv_seed is not None else DEFAULT_CV_SEED
            folds = [FoldRows(fold, cv, fold_seed) for fold in range(cv)]
        design, *fold_designs = run.learn(folds)
        builder = _training_builder(
            pl.DataFrame(schema=self.source.schema), design, np.empty((0, design.p)), []
        )
        builder._last_names = list(design.names)
        self._builder = builder
        self.feature_names = list(design.names)

        # 2. The solve.
        from rustystats.formula import (
            SmoothTermResult,
            _get_constraint_indices,
            _monotone_smooth_endpoint_constraints,
        )

        smooth_terms, smooth_col_indices = builder.get_smooth_terms()
        if smooth_terms and alpha > 0.0 and l1_ratio > 0.0:
            raise ValidationError(_SMOOTH_L1)
        nonneg, nonpos = _get_constraint_indices(self.feature_names)
        if nb_estimate and (smooth_terms or nonneg or nonpos):
            kinds = "/".join(
                kind
                for kind, on in (("smooth", smooth_terms), ("sign-constrained", nonneg or nonpos))
                if on
            )
            raise ValidationError(_theta_estimate_refusal(kinds))
        if cv is not None and smooth_terms:
            raise ValidationError(_SMOOTH_CV)
        from rustystats.formula import _resolved_max_iter

        max_iter = _resolved_max_iter(max_iter, smooth_terms)
        # The penalty matrices are planned from their dimensions and built only once
        # the solver phases are admitted (§11).
        penalty_dims = [end - start for start, end in smooth_col_indices]
        monotonicity = [
            getattr(term, "_smooth_monotonicity", None) or getattr(term, "monotonicity", None)
            for term in smooth_terms
        ]
        # §6.7: centres and scales are learned state, charged before the moments.
        standardized = standardize and not smooth_terms and (cv is not None or alpha > 0.0)
        if standardized:
            copies = 1 + (len(fold_designs) if te_cv else 0)
            run.charge_learned(3 * 8 * design.p * copies, term="standardization", count=design.p)
        run.plan_solve(
            design,
            robust=len(robust_types or []),
            penalty_dims=penalty_dims,
            monotone=any(m is not None for m in monotonicity),
            folds=cv if cv is not None else 0,
            lockstep=cv is not None and not te_cv,
            standardized=standardized,
            theta=nb_estimate,
        )
        # With target-encoded CV, each fold's passes hold their own encode block; the
        # full design's is allocated once the folds are done.
        open_pass = None if te_cv else run.solver_passes(design)
        theta_value = nb_theta if nb_theta is not None else DEFAULT_NEGBINOMIAL_THETA
        requested_alpha, requested_l1 = alpha, l1_ratio
        path_info = None
        if cv is not None:
            if te_cv:
                path_info = self._te_cv_path(
                    run,
                    design,
                    fold_designs,
                    cv=cv,
                    selection=selection,
                    regularization=regularization,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                    l1_ratio=l1_ratio,
                    max_iter=cv_max_iter if cv_max_iter is not None else max_iter,
                    tol=cv_tol if cv_tol is not None else tol,
                    seed=folds[0].seed,
                    include_unregularized=include_unregularized,
                    verbose=verbose,
                    standardize=standardize,
                    theta=theta_value,
                )
                # The refit's design sorts its training values after the folds' are
                # gone (§10).
                run.learn_values(design)
                open_pass = run.solver_passes(design)
            else:
                path_info = self._cv_path(
                    run,
                    design,
                    open_pass,
                    cv=cv,
                    selection=selection,
                    regularization=regularization,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                    l1_ratio=l1_ratio,
                    max_iter=cv_max_iter if cv_max_iter is not None else max_iter,
                    tol=cv_tol if cv_tol is not None else tol,
                    seed=cv_seed,
                    include_unregularized=include_unregularized,
                    verbose=verbose,
                    standardize=standardize,
                    theta=theta_value,
                    nonneg=nonneg,
                    nonpos=nonpos,
                )
            alpha, l1_ratio = path_info.selected_alpha, path_info.selected_l1_ratio
        smooth_results = total_edf = gcv = None
        theta_metadata = None
        if nb_estimate:
            result, theta_metadata = self._fit_theta(
                run, design, open_pass, max_iter, tol, robust_types
            )
            nb_theta = float(theta_metadata["theta"])
        elif smooth_terms:
            # Admitted with the solve phase (plan_solve), built now.
            penalties = [
                term.compute_penalty_matrix(k)[:k, :k]
                for term, k in zip(smooth_terms, penalty_dims)
            ]
            nonneg_s, nonpos_s = _monotone_smooth_endpoint_constraints(
                smooth_terms,
                smooth_col_indices,
                nonneg if nonneg else None,
                nonpos if nonpos else None,
            )
            lambda_min, lambda_max = (
                (alpha, alpha) if alpha > 0.0 else (DEFAULT_LAMBDA_MIN, DEFAULT_LAMBDA_MAX)
            )
            result, meta = fit_smooth_glm_batched_py(
                open_pass,
                run.n_rows,
                design.p,
                run.y_mean,
                smooth_col_indices,
                penalties,
                self.family,
                link=self.link,
                has_offset=run.has_offset,
                max_iter=max_iter,
                tol=tol,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                smooth_monotonicity=(
                    monotonicity if any(m is not None for m in monotonicity) else None
                ),
                nonneg_indices=nonneg_s,
                nonpos_indices=nonpos_s,
                var_power=self.var_power,
                theta=theta_value,
                allow_extended_tweedie=self.allow_extended_tweedie,
                robust=robust_types,
                **run.kernel_kwargs(design),
            )
            smooth_results = []
            for i, (term, (start, end)) in enumerate(zip(smooth_terms, smooth_col_indices)):
                smooth_results.append(
                    SmoothTermResult(
                        variable=term.var_name,
                        k=term.df,
                        edf=meta["smooth_edfs"][i],
                        lambda_=meta["lambdas"][i],
                        gcv=meta["gcv"],
                        col_start=start,
                        col_end=end,
                    )
                )
                term._lambda = meta["lambdas"][i]
                term._edf = meta["smooth_edfs"][i]
            total_edf, gcv = meta["total_edf"], meta["gcv"]
        else:
            center = scale = None
            if standardize and alpha > 0.0:
                if path_info is not None and path_info.final_fit_center is not None:
                    center, scale = path_info.final_fit_center, path_info.final_fit_scale
                else:
                    # Recomputed for the refit after CV with target encoding, as
                    # in memory (formula.py `_fit_glm_core`).
                    center, scale = run.standardization(design, self.intercept, open_pass)
            result = fit_glm_batched_py(
                open_pass,
                run.n_rows,
                design.p,
                run.y_mean,
                self.family,
                link=self.link,
                var_power=self.var_power,
                theta=theta_value,
                has_offset=run.has_offset,
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                nonneg_indices=nonneg if nonneg else None,
                nonpos_indices=nonpos if nonpos else None,
                allow_extended_tweedie=self.allow_extended_tweedie,
                fit_intercept=self.intercept,
                center=center,
                scale=scale,
                skip_covariance=not compute_covariance,
                robust=robust_types,
                **run.kernel_kwargs(design),
            )
        return self._results(
            result,
            alpha=alpha,
            l1_ratio=l1_ratio,
            standardize=standardize,
            compute_covariance=compute_covariance,
            smooth_results=smooth_results,
            total_edf=total_edf,
            gcv=gcv,
            theta=nb_theta,
            theta_metadata=theta_metadata,
            max_iter=max_iter,
            tol=tol,
            run=run,
            requested=(requested_alpha, requested_l1, cv, regularization, path_info),
        )

    def _cv_path(
        self,
        run: _Run,
        design: FrozenDesign,
        open_pass: Callable,
        *,
        cv: int,
        selection: str,
        regularization: str,
        n_alphas: int,
        alpha_min_ratio: float,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        seed: int | None,
        include_unregularized: bool,
        verbose: bool,
        standardize: bool,
        theta: float,
        nonneg: list[int],
        nonpos: list[int],
    ):
        """The CV path over batches (§7.8): ``fit_cv_regularization_path``'s steps."""
        from rustystats._rustystats import alpha_max_batched_py, fit_cv_path_batched_py
        from rustystats.constants import DEFAULT_CV_SEED, DEFAULT_ELASTIC_NET_L1_RATIO
        from rustystats.regularization_path import (
            _alpha_link,
            cv_path_info,
            generate_alpha_path,
            penalized_column_mask,
            solver_standardization,
        )

        if regularization == "ridge":
            effective_l1 = 0.0
        elif regularization == "lasso":
            effective_l1 = 1.0
        else:
            effective_l1 = l1_ratio if l1_ratio > 0 else DEFAULT_ELASTIC_NET_L1_RATIO
        p = design.p
        fit_intercept = bool(self.intercept)
        mask = penalized_column_mask(p, fit_intercept)
        center = scale = ridge = None
        if standardize:
            center, scale, ridge = compute_standardization_batched_py(
                open_pass,
                run.n_rows,
                p,
                weighted=True,
                pen_mask=[bool(v) for v in mask],
                fit_intercept=fit_intercept,
                **run.threads_kwargs,
            )
            center, scale, ridge = np.asarray(center), np.asarray(scale), np.asarray(ridge)
        link = _alpha_link(self.link, self.family)
        trace: list = []
        try:
            alpha_max = alpha_max_batched_py(
                open_pass,
                run.n_rows,
                p,
                run.y_mean,
                self.family,
                link,
                effective_l1,
                has_offset=run.has_offset,
                var_power=self.var_power,
                theta=theta,
                pen_mask=[bool(v) for v in mask],
                center=center,
                scale=scale,
                ridge_xtx_diag=None if ridge is None else [float(v) for v in ridge],
                allow_extended_tweedie=self.allow_extended_tweedie,
                trace=trace,
                **run.kernel_kwargs(design),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        alphas = generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio)
        if include_unregularized and (alphas.size == 0 or not np.any(alphas == 0.0)):
            alphas = np.concatenate([alphas, np.array([0.0], dtype=np.float64)])
        solver_center, solver_scale = solver_standardization(
            center, scale, fit_intercept=fit_intercept
        )
        rust_result = fit_cv_path_batched_py(
            open_pass,
            run.n_rows,
            p,
            run.y_mean,
            self.family,
            link=self.link,
            var_power=self.var_power,
            theta=theta,
            has_offset=run.has_offset,
            weighted=self._weights_spec is not None,
            alphas=list(alphas),
            l1_ratio=effective_l1,
            n_folds=cv,
            max_iter=max_iter,
            tol=tol,
            seed=seed if seed is not None else DEFAULT_CV_SEED,
            nonneg_indices=nonneg if nonneg else None,
            nonpos_indices=nonpos if nonpos else None,
            allow_extended_tweedie=self.allow_extended_tweedie,
            fit_intercept=fit_intercept,
            center=solver_center,
            scale=solver_scale,
            solve_workspaces=run.solve_workspaces,
            **run.kernel_kwargs(design),
        )
        return cv_path_info(
            rust_result,
            l1_ratio=effective_l1,
            selection=selection,
            n_folds=cv,
            n_params=p,
            cv_max_iter=max_iter,
            cv_tol=tol,
            final_fit_center=solver_center,
            final_fit_scale=solver_scale,
            verbose=verbose,
            solver_trace=trace,
        )

    def _te_cv_path(
        self,
        run: _Run,
        design: FrozenDesign,
        fold_designs: list[FrozenDesign],
        *,
        cv: int,
        selection: str,
        regularization: str,
        n_alphas: int,
        alpha_min_ratio: float,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        seed: int,
        include_unregularized: bool,
        verbose: bool,
        standardize: bool,
        theta: float,
    ):
        """CV with target encoding over batches (§7.9): ``fit_cv_te_regularization_path``'s
        steps, each fold's passes encoding every row with the fold's design."""
        from rustystats.regularization_path import target_encoded_cv_path

        if regularization == "elastic_net":
            l1_ratio = l1_ratio if l1_ratio > 0 else 0.5
        folds = _BatchedTargetFolds(run, fold_designs, seed, theta)
        return target_encoded_cv_path(
            folds,
            n_folds=cv,
            n_params=design.p,
            family=self.family,
            link=self.link,
            regularization=regularization,
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
            tol=tol,
            include_unregularized=include_unregularized,
            standardize=standardize,
            fit_intercept=bool(self.intercept),
            selection=selection,
            verbose=verbose,
        )

    def _resolve_theta(
        self, alpha: float, cv: int | None, regularization: str | None, options: OutOfCoreOptions
    ) -> tuple[bool, float | None]:
        """Whether θ is estimated, and the fixed θ (None for other families)."""
        from rustystats.formula import is_negbinomial_family

        if not is_negbinomial_family(self.family):
            return False, None
        if self.theta is None:
            raise ValidationError(
                "Negative Binomial requires an explicit theta. Pass a positive numeric "
                "theta= for a fixed-dispersion fit, or theta='estimate' to opt in to "
                "profile-likelihood estimation on the plain GLM path."
            )
        if isinstance(self.theta, str):
            if self.theta != "estimate":
                raise ValidationError(
                    f"theta must be a positive number or 'estimate', got {self.theta!r}."
                )
            if regularization is not None or cv is not None or alpha != 0.0:
                raise ValidationError(
                    "Negative Binomial theta estimation is not supported for regularized "
                    "models (the profile loop is defined only for the plain GLM path). Pass "
                    "an explicit numeric theta= (e.g. theta=1.0) for these models."
                )
            if options.scratch_dir is None:
                raise OutOfCoreOptionError(
                    "Negative Binomial θ estimation keeps μ in a spill file between rounds; "
                    "set OutOfCoreOptions.scratch_dir."
                )
            return True, None
        if self.theta <= 0:
            raise ValidationError(f"theta must be > 0 for negative binomial, got {self.theta}.")
        return False, float(self.theta)

    def _fit_theta(
        self,
        run: _Run,
        design: FrozenDesign,
        open_pass: Callable,
        max_iter: int,
        tol: float,
        robust_types: list[str] | None,
    ) -> tuple[Any, dict]:
        """θ by profile likelihood over passes (§7.10), μ spilled under scratch_dir."""
        from rustystats._rustystats import fit_negbinomial_batched_py

        spill = _MuSpill(run, self.options_scratch_dir(run))
        try:
            result, meta = fit_negbinomial_batched_py(
                open_pass,
                spill,
                spill.open,
                run.n_rows,
                design.p,
                run.y_mean,
                link=self.link or "log",
                init_theta=None,
                theta_tol=1e-5,
                max_theta_iter=25,
                has_offset=run.has_offset,
                weighted=self._weights_spec is not None,
                max_iter=max_iter,
                tol=tol,
                robust=robust_types,
                **run.kernel_kwargs(design),
            )
        finally:
            spill.remove()
        return result, dict(meta)

    @staticmethod
    def options_scratch_dir(run: _Run) -> str:
        return os.fspath(run.options.scratch_dir)

    def _results(
        self,
        result: Any,
        *,
        alpha: float,
        l1_ratio: float,
        standardize: bool,
        compute_covariance: bool,
        smooth_results: list | None,
        total_edf: float | None,
        gcv: float | None,
        theta: float | None,
        theta_metadata: dict | None,
        max_iter: int,
        tol: float,
        run: _Run,
        requested: tuple,
    ) -> GLMModel:
        from rustystats.formula import (
            FormulaGLMDict,
            _build_results,
            _format_result_family,
            _solver_settings,
        )

        result_family = _format_result_family(self.family, self.var_power, theta)
        covariance_available = bool(
            compute_covariance or theta_metadata is not None or smooth_results is not None
        )
        results = _build_results(
            result,
            self.feature_names,
            self.formula,
            result_family,
            self.link,
            self._builder,
            # §4.5: column specs are kept; an array the caller passed is not, and
            # prediction must be given one for the new data.
            self._offset_spec
            if self._offset is None or self._offset.column is not None
            else _array_required("offset"),
            self._exposure_spec
            if self._exposure is None or self._exposure.column is not None
            else None,
            requested[4],
            smooth_results,
            total_edf,
            gcv,
            terms_dict=self.terms,
            interactions_spec=self.interactions_spec,
            input_transforms=self._input_transforms,
            compiled_input_transforms=self._compiled_input_transforms,
            complement_spec=self._complement_spec,
            # No per-row complement values are kept (§4.5): has_complement reads
            # `_fitted_with_complement` instead.
            complement_values=None,
            array_exposure_requires_prediction_override=(
                self._exposure is not None and self._exposure.column is None
            ),
            regularization_standardized=bool(standardize and alpha > 0.0),
            covariance_available=covariance_available,
        )
        results._fitted_with_complement = self._complement is not None
        results.theta = theta
        results.theta_metadata = theta_metadata or (
            {
                "estimated": False,
                "theta": float(theta),
                "init_theta": None,
                "theta_iterations": 0,
                "theta_converged": None,
                "theta_tol": None,
                "max_theta_iter": None,
                "glm_tol": float(tol),
                "fallback_reason": None,
            }
            if theta is not None
            else None
        )
        results.inference_status, results.optimizer_route = (
            FormulaGLMDict._inference_status_and_route(self, *requested)
        )
        if not covariance_available and not str(results.inference_status).startswith("naive_"):
            results.inference_status = "covariance_skipped"
        results.solver_status = getattr(result, "solver_status", "converged")
        results.step_halving_used = bool(getattr(result, "step_halving_used", False))
        results.stationary = getattr(result, "stationary", None)
        results.max_std_score = getattr(result, "max_std_score", None)
        # A weights column is kept for diagnostics; an array is not (§4.5).
        results._weights_spec = self._weights_spec if isinstance(self._weights_spec, str) else None
        results.memory_plan = run.memory_plan
        results.encode_rows = run.encode_rows
        # §4.5: the I/O passes per pass kind, and the caps the solve used.
        results.fit_mode = "out_of_core"
        results.passes = dict(run.pass_counts)
        results.solver_settings = _solver_settings(
            max_iter=max_iter,
            tol=tol,
            target_encoding_seed=run.te_seed,
            theta_metadata=results.theta_metadata,
        )
        # The drivers' steps: alpha_max and CV's, then the fit's (§4.5).
        results.solver_trace = solver_steps(
            getattr(requested[4], "solver_trace", ()),
            getattr(result, "solver_trace", ()),
        )
        return results


def _array_required(name: str) -> Any:
    from rustystats.formula import ArrayInputRequired

    return ArrayInputRequired(name)


_SMOOTH_L1 = (
    "L1/elastic-net regularization is not supported for smooth terms. "
    "Use l1_ratio=0 for the fixed spline penalty path, or specify "
    "df=/knots= for unpenalized fixed basis columns."
)
_SMOOTH_CV = (
    "CV-based regularization is not supported for smooth terms yet because "
    "the path solver uses scalar penalties while smooth fits use spline "
    "difference penalties. Pass a numeric alpha with l1_ratio=0, or use "
    "df=/knots= for fixed basis columns."
)


def _theta_estimate_refusal(kinds: str) -> str:
    return (
        "Negative Binomial theta estimation is not supported for "
        f"{kinds} models (the profile loop is defined only for the plain GLM path). "
        "Pass an explicit numeric theta= (e.g. theta=1.0) for these models."
    )


# ── The Negative Binomial μ spill (§10) ─────────────────────────────────────


class _MuSpill:
    """μ of every row, written by Rust a block at a time and read back for θ passes.

    Rust calls the spill with ``None`` to start it over (each round), then with each
    block of μ. :meth:`open` yields ``(y, weights, mu)`` blocks for one θ pass,
    reading y and the weights from the source and μ from the file in step.
    """

    def __init__(self, run: _Run, scratch_dir: str) -> None:
        import tempfile

        self._run = run
        try:
            handle, self.path = tempfile.mkstemp(prefix="rustystats-nb-mu-", dir=scratch_dir)
            os.close(handle)
        except OSError as exc:
            raise SpillError(f"Cannot create a spill file in {scratch_dir!r}: {exc}") from exc
        self._file = None

    def __call__(self, block: np.ndarray | None) -> None:
        try:
            if block is None:
                # Each round's spill follows a θ pass, which closed the last one.
                self._file = open(self.path, "wb")  # noqa: SIM115
            else:
                np.asarray(block, dtype=np.float64).tofile(self._file)
        except OSError as exc:
            raise SpillError(f"Cannot write the spill file {self.path!r}: {exc}") from exc

    def open(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        run = self._run
        model = run.model
        if self._file is not None:
            self._file.close()
            self._file = None
        weights = model._weights_spec
        columns = [model.response] + ([weights] if isinstance(weights, str) else [])
        if not set(columns) <= set(model.source.schema) or model._compiled_input_transforms:
            columns = model._source_columns
        try:
            spilled = open(self.path, "rb")  # noqa: SIM115
        except OSError as exc:
            raise SpillError(f"Cannot read the spill file {self.path!r}: {exc}") from exc
        with spilled:
            row = 0
            for rows in run.slices("theta", columns, run.encode_rows, run.payload_bound(columns)):
                y, weights_block, _offset = run._row_arrays(rows, row, offset=False)
                mu = np.fromfile(spilled, dtype=np.float64, count=rows.height)
                if mu.shape[0] != rows.height:
                    raise BatchSourceChangedError(
                        "The source yielded more rows on a θ pass than were spilled: "
                        "the source changed between passes."
                    )
                row += rows.height
                yield y, weights_block, mu

    def remove(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise SpillError(f"Cannot remove the spill file {self.path!r}: {exc}") from exc


# ── One fit's passes ────────────────────────────────────────────────────────


def _learn_slice_bytes(max_bytes: int | None) -> int:
    """A learning slice's byte bound (§4.3): 64 MiB, less under ``max_bytes``."""
    if max_bytes is None:
        return DEFAULT_LEARN_SLICE_BYTES
    return max(1, min(DEFAULT_LEARN_SLICE_BYTES, max_bytes // _LEARN_SLICE_SHARE))


def _learning_peak(learn_slice_bytes: int) -> int:
    """A learning pass's working set (§4.3, §11): the copies of one slice's payload
    alive at once, and its per-row scratch."""
    return _LEARN_PAYLOAD_COPIES * learn_slice_bytes + E_LEARN * _LEARN_ROW_BYTES


class _Run:
    """The passes of one out-of-core fit: their progress, fingerprints and memory."""

    def __init__(self, model: OutOfCoreGLMDict, options: OutOfCoreOptions) -> None:
        self.model = model
        self.options = options
        self.pass_index = -1
        self.pass_counts: dict[str, int] = {}
        self.n_rows: int | None = None
        self.y_mean = 1.0
        self.has_offset = (
            model._exposure_spec is not None
            or model._offset_spec is not None
            or model._complement is not None
        )
        self.fingerprints: dict[tuple[str, str], tuple[int, int]] | None = None
        self.budget = LearnedStateBudget(options.max_learned_state_bytes)
        self.threads = options.threads or _pool_threads()
        self.threads_kwargs: dict[str, int] = (
            {} if options.threads is None else {"threads": int(options.threads)}
        )
        #: Lockstep CV folds solving between passes at once (each 2·8p², §7.8).
        self.solve_workspaces = 1
        self.learn_slice_bytes = _learn_slice_bytes(options.max_bytes)
        #: The largest row a learning pass found larger than a slice (§4.3).
        self.max_row_bytes = 0
        self.encode_rows = 0
        self.spill_bytes = 0
        self.memory_plan: MemoryPlan | None = None
        self._phases: dict[str, int] = {}
        #: Persistent bytes besides learned state: the allowance, the fingerprints and,
        #: once p is known, the drivers' state.
        self._persistent = _ALLOCATOR_ALLOWANCE + 64 * len(model._source_columns)
        #: Target encoding: the seed (drawn once when the model has none, DEC-3),
        #: records per sort run, the planned spill peak and the inputs.
        self.te_seed: int | None = None
        self.te_run_records = 0
        self.te_spill_bytes = 0
        self._target: TargetInputs | None = None
        self._open_learning: Callable[[str], Iterator[pl.DataFrame]] | None = None
        #: Spill files the fit owns until it ends, and the designs reading them.
        self.spill_paths: set[str] = set()
        self.designs: list[FrozenDesign] = []
        self._stage_index = -1

    def kernel_kwargs(self, design: FrozenDesign) -> dict[str, int]:
        """A fit binding's thread count and the design's structural nonzeros per row,
        from which it chooses dense or sparse rows (§5.2)."""
        return {"max_row_nonzeros": design.max_row_nonzeros, **self.threads_kwargs}

    @property
    def persistent_bytes(self) -> int:
        """Everything alive between phases: the fixed persistent state and the learned
        state counted so far (§11)."""
        return self._persistent + int(self.budget.used)

    # ── Source passes ──────────────────────────────────────────────────────

    def payload_bound(self, columns: Sequence[str]) -> int | None:
        """A solver pass's slices are bounded by their variable-width payload, so the
        copies made of it (keys, fingerprints) stay within the plan; None when every
        column read is fixed-width."""
        schema = self.model.source.schema
        variable = any(schema[column] == kind for column in columns for kind in _VARIABLE_WIDTH)
        return self.learn_slice_bytes if variable else None

    def slices(
        self,
        kind: str,
        columns: Sequence[str],
        max_rows: int,
        max_bytes: int | None = None,
        *,
        fixed_bytes: int = 0,
        learning: bool = False,
    ) -> Iterator[pl.DataFrame]:
        """One pass over the source: every batch cut into slices of at most
        ``max_rows`` rows and ``max_bytes`` payload bytes, each fingerprinted and
        input-transformed on its own, so no work spans a whole source batch (§4.3).

        In a learning pass a row larger than a slice is charged before anything is
        done with it. The caller's arrays (weights, offset, exposure, complement) are
        read for each slice by global row and fingerprinted beside the columns.
        ``on_batch`` runs after each source batch. The fingerprints are compared with
        the freeze pass's when the pass ends (§9).
        """
        self.pass_index += 1
        self.pass_counts[kind] = self.pass_counts.get(kind, 0) + 1
        index = self.pass_index
        arrays = self.model._array_inputs
        # Tagged keys: a column named like an array's key never shares its fingerprint.
        prints = {("column", column): ColumnFingerprint() for column in columns}
        prints.update({row_input.key: ColumnFingerprint() for row_input in arrays})
        compiled = self.model._compiled_input_transforms
        rows_done = 0
        admit = self._admit_oversized if learning else None
        for batch in self.model.source.batches(list(columns)):
            for rows in slice_rows(
                batch, max_rows, max_bytes, fixed_bytes=fixed_bytes, admit=admit
            ):
                for column in columns:
                    fingerprint_update(prints[("column", column)], rows[column])
                for row_input in arrays:
                    values = row_input.prepare(rows_done, rows.height)
                    fingerprint_update(prints[row_input.key], pl.Series(values))
                rows_done += rows.height
                yield apply_input_transforms(rows, compiled) if compiled else rows
            if self.options.on_batch is not None:
                self.options.on_batch(BatchProgress(index, kind, rows_done, self.n_rows))
        digests = {key: (fp.rows, fp.digest()) for key, fp in prints.items()}
        if self.fingerprints is None:
            self.fingerprints = digests
            return
        for (source, name), digest in digests.items():
            if self.fingerprints.get((source, name), digest) != digest:
                what = f"the {name} array" if source == "array" else repr(name)
                raise BatchSourceChangedError(
                    f"Pass {index} ({kind}) read {what} differently from the freeze pass: "
                    "the source changed between passes."
                )

    def _admit_oversized(self, column: str, row_bytes: int) -> None:
        """Charge a row larger than a learning slice against the learning budget before
        it is sliced out, fingerprinted, transformed or learned (§4.3); the solver
        passes plan the largest such row's copies (:meth:`plan_solve`)."""
        self.budget.admit_row(
            column, row_bytes, transient=_OVERSIZED_ROW_COPIES * row_bytes + _LEARN_ROW_BYTES
        )
        self.max_row_bytes = max(self.max_row_bytes, row_bytes)

    # ── Learning ───────────────────────────────────────────────────────────

    def learn(self, folds: Sequence[FoldRows] = ()) -> list[FrozenDesign]:
        """The learning passes (§6.2): freeze, knots, and target encoding's te_stats and
        te_sort passes with their spill stages; the response mean and whole-column
        checks. Returns the full data's design, then one per fold of ``folds`` learned
        from its training rows (cross-validation with target encoding, §7.9); the full
        design's training values then wait for :meth:`learn_values`."""
        from rustystats.regularization_path import fold_formulas

        model = self.model
        facts = _FreezeFacts()
        all_columns = model._source_columns
        spline_vars = {spline.var_name for spline in _spline_terms(model._parsed)}
        knot_columns = sorted(
            (
                (spline_vars | input_transform_source_columns(model._input_transforms))
                & set(model.source.schema)
            )
            or set(all_columns[:1])
        )
        target_columns = self._target_columns()

        def open_slices(kind: str) -> Iterator[pl.DataFrame]:
            freeze = kind == "freeze"
            if not freeze and self.n_rows is None:
                # The freeze pass has ended: its whole-column checks come first.
                self._finish_freeze(facts)
            columns = all_columns if freeze else knot_columns if kind == "knots" else target_columns
            row = 0
            for rows in self.slices(
                kind, columns, E_LEARN, self.learn_slice_bytes, fixed_bytes=8, learning=True
            ):
                if freeze:
                    self._observe(rows, row, facts)
                row += rows.height
                yield rows

        learning = _learning_peak(self.learn_slice_bytes)
        self._admit("freeze", learning)
        if self.options.max_bytes is not None:
            # §6.4: learned state also fits what max_bytes leaves beside the learning
            # passes' working set, whether or not max_learned_state_bytes is set.
            other = self.persistent_bytes + learning
            self.budget.cap(
                self.options.max_bytes - other,
                source=f"max_bytes = {self.options.max_bytes} (less {other:,} bytes of other state)",
            )
        self._open_learning = open_slices
        if model._parsed.target_encoding_terms:
            self._target = self._target_inputs()
        parsed = [model._parsed, *fold_formulas(model, len(folds))]
        try:
            designs = learn_frozen_designs(
                parsed,
                open_slices,
                [None, *folds],
                budget=self.budget,
                target=self._target,
                values=[not folds] + [True] * len(folds),
                max_design_columns=self.options.max_design_columns,
                admit=lambda phase, extra: self._admit(phase, learning + extra),
            )
        except FrozenDesignUnsupported as exc:
            if self.n_rows is None and facts.rows == 0:
                # An empty source: refused as in memory, for its empty response.
                self._finish_freeze(facts)
            raise OutOfCoreOptionError(
                f"This model cannot be learned out of core: {exc}. Fit it in memory."
            ) from exc
        self.designs.extend(designs)
        if self.n_rows is None:
            self._finish_freeze(facts)
        self.te_spill_bytes = self._target_spill_bytes(designs)
        return designs

    def learn_values(self, design: FrozenDesign) -> None:
        """Sort a design's target-encoding training values (te_sort and the spill
        stages), when :meth:`learn` deferred them."""
        assert self._target is not None and self._open_learning is not None
        learn_target_values([design], self._open_learning, self._target)

    def charge_learned(self, nbytes: int, *, term: str, count: int) -> None:
        """Charge learned state the solve keeps (centres and scales, §6.7) to the
        learned-state budget, before it is allocated."""
        self.budget.charge(nbytes, term=term, cardinality=count)

    # ── Target encoding (§6.6, §10) ────────────────────────────────────────

    def _target_columns(self) -> list[str]:
        """Columns the te_stats and te_sort passes read."""
        model = self.model
        if model._compiled_input_transforms:
            return model._source_columns
        columns = {model.response} | {
            variable
            for term in model._parsed.target_encoding_terms
            for variable in _target_variables(term)
        }
        if isinstance(model._exposure_spec, str):
            columns.add(model._exposure_spec)
        return sorted(columns)

    def _target_inputs(self) -> TargetInputs:
        """Target encoding's inputs: the response and exposure, the seed, and sort runs
        spilled under ``scratch_dir`` (§6.6), sized once the freeze pass has counted
        the learned state (§11)."""
        model = self.model
        self.te_seed = model._seed if model._seed is not None else draw_target_encoding_seed()
        exposure = model._exposure
        scratch_dir = self.options.scratch_dir
        assert scratch_dir is not None
        return TargetInputs(
            target=lambda rows, first_row: _float64(rows, model.response),
            exposure=None if exposure is None else exposure.read,
            seed=self.te_seed,
            scratch_dir=os.fspath(scratch_dir),
            run_records=self._te_sort_records,
            progress=self.spill_progress,
            keep=self.spill_paths.add,
        )

    def _te_sort_records(self) -> int:
        """Records per sort run, admitted beside the learned state counted so far.

        The default sort budget is the encode slice's; under ``max_bytes`` it is what
        remains beside the persistent and learned state and the larger of a learning
        slice (te_sort) and the merge buffers (te_merge, te_walk) (§11).
        """
        learning = _learning_peak(self.learn_slice_bytes)
        beside = max(learning, _TE_MERGE_BYTES)
        if self.options.te_run_records is not None:
            run_records = self.options.te_run_records
        else:
            budget = DEFAULT_ENCODE_BLOCK_BYTES
            if self.options.max_bytes is not None:
                budget = min(budget, self.options.max_bytes - self.persistent_bytes - beside)
            run_records = budget // _TE_RECORD_PLAN_BYTES
            if run_records < 1:
                raise MemoryPlanExceededError(
                    f"Target encoding's sort needs at least {_TE_RECORD_PLAN_BYTES} bytes per "
                    f"record beside {beside:,} bytes of merge buffers or learning slice and "
                    f"{self.persistent_bytes:,} persistent bytes; max_bytes = "
                    f"{self.options.max_bytes} leaves none."
                )
        records = run_records * _TE_RECORD_PLAN_BYTES
        self._admit("te_sort", records + learning)
        self._admit("te_merge", records + _TE_MERGE_BYTES)
        self.te_run_records = run_records
        return run_records

    def _target_spill_bytes(self, designs: list[FrozenDesign]) -> int:
        """The planned spill peak of target encoding (§10)."""
        terms = self.model._parsed.target_encoding_terms
        if not terms:
            return 0
        t = len(terms)
        m = max(design_target.n_permutations for design_target in designs[0].target)

        def one(rows: int) -> int:
            return 8 * rows * (t + m - 2) + 80 * rows

        peak = one(designs[0].training_rows)
        earlier = 0
        for fold_design in designs[1:]:
            peak = max(peak, earlier + one(fold_design.training_rows))
            earlier += 8 * t * fold_design.training_rows
        return peak

    def spill_progress(self, kind: str, done: int, total: int, start: bool) -> None:
        """A spill stage's progress (§4.4): each stage is a pass of its own, and
        ``on_batch`` hears records done of the stage's total."""
        if start:
            self.pass_index += 1
            self.pass_counts[kind] = self.pass_counts.get(kind, 0) + 1
            self._stage_index = self.pass_index
            return
        if self.options.on_batch is not None:
            self.options.on_batch(BatchProgress(self._stage_index, kind, done, total))

    def release_design(self, design: FrozenDesign) -> None:
        """A design's passes have ended: close its columns and remove its spill files."""
        design.close()
        for frozen in design.target:
            # A fold design's training values are always spilled (out of core).
            self.spill_paths.discard(frozen.values.path)
            _remove_spill(frozen.values.path)

    def close(self, quiet: bool = False) -> None:
        """End the fit: close every design and remove every spill file. A file that
        cannot be removed raises :class:`SpillError` unless ``quiet`` (another
        exception is already on its way out)."""
        for design in self.designs:
            design.close()
        failure: SpillError | None = None
        for path in sorted(self.spill_paths):
            try:
                _remove_spill(path)
            except SpillError as exc:
                failure = failure or exc
        self.spill_paths.clear()
        if failure is not None and not quiet:
            raise failure

    def _observe(self, rows: pl.DataFrame, start: int, facts: _FreezeFacts) -> None:
        """The freeze pass's per-slice checks, in the in-memory fit's order: exposure,
        the complement's domain, y, weights, then the combined link-scale offset
        (finite, as ``validate_offset`` requires of it in memory)."""
        model = self.model
        where = f" (source rows {start}..{start + rows.height - 1})"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if model._exposure is not None:
                    exposure = model._exposure.read(rows, start)
                    if not np.all(np.isfinite(exposure)):
                        raise ValidationError("exposure must be finite.")
                    if np.any(exposure <= 0):
                        raise ValidationError("exposure must be strictly positive.")
                offset = self._link_offset(rows, start)
                y = _float64(rows, model.response)
                validate_response(
                    y,
                    model.family,
                    require_variation=False,
                    var_power=model.var_power,
                    allow_extended_tweedie=model.allow_extended_tweedie,
                )
                if model._weights is not None:
                    w = coerce_to_float64(model._weights.read(rows, start), "weights")
                    if np.any(w < 0):
                        validate_weights(w, len(w))
                    facts.weight_sum += float(np.sum(w))
                    facts.zero_weights += int(np.sum(w == 0))
                if offset is not None:
                    validate_offset(offset, len(y), model.family)
            except ValidationError as exc:
                raise ValidationError(f"{exc}{where}") from exc
        # A slice is never empty.
        if facts.first_response is None:
            facts.first_response = float(y[0])
        if not facts.response_varies and np.any(y != facts.first_response):
            facts.response_varies = True
        facts.response_sum.update(np.ascontiguousarray(y))
        facts.rows += len(y)

    def _finish_freeze(self, facts: _FreezeFacts) -> None:
        model = self.model
        self.n_rows = facts.rows
        if facts.rows == 0:
            raise ValidationError("response (y) is empty. Cannot fit model with no observations.")
        for row_input in (model._exposure, model._offset, model._weights, model._complement_input):
            if row_input is not None:
                row_input.check_rows(facts.rows)
        if not facts.response_varies:
            raise ValidationError(
                f"response (y) is constant (all values = {facts.first_response}). "
                "A GLM requires variation in the response variable."
            )
        if model._weights is not None and facts.weight_sum == 0:
            raise ValidationError(
                "weights sum to zero. At least some observations must have positive weight."
            )
        self.y_mean = facts.response_sum.mean()

    # ── Memory plan (§11) ──────────────────────────────────────────────────

    def _admit(self, phase: str, peak: int) -> None:
        """Admit a phase's peak beside the persistent and learned state before the
        phase allocates anything."""
        peak = int(peak)
        self._phases[phase] = max(self._phases.get(phase, 0), peak)
        limit = self.options.max_bytes
        persistent = self.persistent_bytes
        if limit is not None and persistent + peak > limit:
            raise MemoryPlanExceededError(
                f"The {phase} phase plans a peak of {persistent + peak:,} bytes "
                f"({peak:,} for the phase, {persistent:,} persistent), more than "
                f"max_bytes = {limit:,}."
            )

    def _per_encode_row(self, design: FrozenDesign, monotone: bool) -> int:
        """Bytes per encoded row in a solver phase (§11): the block and its finiteness
        mask, the monotone block-local copy, the encoder's scratch and the work
        vectors."""
        p = design.p
        block = (8 + _FINITE_MASK_BYTES) * p
        return (
            block + (8 * p if monotone else 0) + design.scratch_bytes_per_row + _WORK_BYTES_PER_ROW
        )

    def _choose_encode_rows(self, per_row: int, fixed: int) -> int:
        """E (§4.3): the largest multiple of GRAM_BLOCK_ROWS whose block fits 256 MiB
        and half of what ``max_bytes`` leaves beside the largest phase's fixed bytes."""
        if self.options.encode_rows is not None:
            return -(-self.options.encode_rows // GRAM_BLOCK_ROWS) * GRAM_BLOCK_ROWS
        budget = DEFAULT_ENCODE_BLOCK_BYTES
        if self.options.max_bytes is not None:
            budget = min(budget, (self.options.max_bytes - self.persistent_bytes - fixed) // 2)
        rows = (budget // max(per_row, 1)) // GRAM_BLOCK_ROWS * GRAM_BLOCK_ROWS
        if rows < GRAM_BLOCK_ROWS:
            raise MemoryPlanExceededError(
                f"Even {GRAM_BLOCK_ROWS} encoded rows of {per_row} bytes do not fit "
                f"max_bytes = {self.options.max_bytes} beside "
                f"{self.persistent_bytes + fixed:,} bytes of persistent and fixed state."
            )
        return rows

    def plan_solve(
        self,
        design: FrozenDesign,
        *,
        robust: int,
        penalty_dims: Sequence[int] = (),
        monotone: bool = False,
        folds: int = 0,
        lockstep: bool = False,
        standardized: bool = False,
        theta: bool = False,
    ) -> None:
        """Plan every solver phase from the frozen design and admit each one (§11),
        before the first of them starts.

        P is one packed Gram with its right-hand side; a block reducer holds T + 2
        partials of each output in a pass (the accumulator, the open block, one per
        thread).
        """
        p = design.p
        partials = self.threads + 2
        gram = 8 * (p * (p + 1) // 2) + 8 * p + 16
        square = 8 * p * p
        driver = _DRIVER_VECTORS * 8 * p
        # Drivers' state lives until the fit ends: one, or one per lockstep fold.
        self._persistent += driver * (1 + (folds if lockstep else 0))
        fixed: dict[str, int] = {}
        # Candidate, trials, Gram and reconstruct passes, then the solves between
        # them: the covariance, a solve workspace and the extraction's system.
        solve = partials * (_GRAMS_IN_FLIGHT * gram + 8 * p) + 3 * square
        if penalty_dims:
            # Smooth fits: each penalty block kept on its term and copied by the
            # binding, the largest block's construction (D, and D'D formed from it),
            # the GCV/REML factorisations, S̃ on the monotone path, the KKT sums and
            # the refresh's score trees (§7.6, §7.7).
            solve += sum(2 * 8 * k * k for k in penalty_dims)
            solve += 3 * 8 * max(penalty_dims) ** 2 + 2 * square
            solve += square if monotone else 0
            solve += partials * 8 * 2 * p + self.threads * _SCORE_TREE_SHIFTS * 8
        fixed["solve"] = solve
        # Pass A: the deviance and statistics, one meat per robust type, with the
        # covariance and the bread the leverages read.
        fixed["final"] = partials * (1 + robust) * gram + 2 * square
        if standardized:
            fixed["moments"] = partials * 8 * (3 * p + 2)
        if folds and lockstep:
            # Every active fold's Gram in the merged pass, the shared solve
            # workspaces, and each fold driver's thread stack (§7.8); α_max's score.
            fixed["cv"] = (
                partials * folds * gram
                + self.solve_workspaces * 2 * square
                + folds * _FOLD_STACK_BYTES
                + partials * 8 * (p + 1)
            )
        elif folds:
            # Folds run one at a time on their own designs (§7.9), each at most p wide.
            fixed["cv"] = solve + partials * 8 * (3 * p + 2)
        if theta:
            fixed["theta"] = solve
        # Sparse rows (§5.2): each reduction thread holds one block's CSR at a time,
        # GRAM_BLOCK_ROWS rows of at most max_row_nonzeros (a u32 index and an f64
        # value each) and the row offsets, sized exactly; plus the packed Gram's row
        # offsets. The moments run on dense rows.
        bound = design.max_row_nonzeros
        if sparse_rows_py(self.n_rows or 0, p, bound):
            block = GRAM_BLOCK_ROWS * 12 * bound + 8 * (GRAM_BLOCK_ROWS + 1)
            csr = self.threads * block + 8 * (p + 1)
            for phase in fixed:
                if phase != "moments":
                    fixed[phase] += csr
        per_row = self._per_encode_row(design, monotone)
        # An encode slice's variable-width payload is bounded by a learning slice, or
        # by the one larger row the freeze pass admitted.
        bound = self.payload_bound(self.model._source_columns)
        payload = _ENCODE_PAYLOAD_COPIES * (max(bound, self.max_row_bytes) if bound else 0)
        theta_rows = _THETA_ROW_BYTES if theta else 0
        largest = max(fixed.values()) + payload
        self.encode_rows = self._choose_encode_rows(per_row + theta_rows, largest)
        encode = self.encode_rows * per_row + payload
        for phase in ("moments", "cv", "solve", "theta", "final"):
            if phase in fixed:
                rows = self.encode_rows * theta_rows if phase == "theta" else 0
                self._admit(phase, encode + rows + fixed[phase])
        self.memory_plan = MemoryPlan(
            phases=dict(self._phases),
            persistent_bytes=self.persistent_bytes,
            spill_bytes=self.te_spill_bytes
            + (8 * (self.n_rows or 0) if self.model._needs_mu_spill else 0),
            encode_rows=self.encode_rows,
            max_bytes=self.options.max_bytes,
            threads=self.threads,
        )

    # ── Solver passes ──────────────────────────────────────────────────────

    def solver_passes(self, design: FrozenDesign) -> Callable[[str], Iterator[tuple]]:
        """``open_pass(kind)`` for the Rust driver: every slice encoded into one block.

        An exception raised while a pass is served travels as :class:`_PassAborted`,
        so no recovery boundary inside the fit mistakes it for a numerical failure;
        :meth:`OutOfCoreGLMDict.fit` re-raises the original (§7.1).
        """
        model = self.model
        buffer = np.empty((self.encode_rows, design.p), dtype=np.float64)
        names = list(design.names)
        columns = model._source_columns
        bound = self.payload_bound(columns)

        def open_pass(kind: str) -> Iterator[tuple]:
            try:
                row = 0
                for rows in self.slices(kind, columns, self.encode_rows, bound):
                    block = buffer[: rows.height]
                    design.encode_into(rows, block, first_row=row)
                    if not np.isfinite(block).all():
                        bad = int(np.flatnonzero(~np.isfinite(block).all(axis=0))[0])
                        raise ValidationError(
                            f"The design column {names[bad]!r} has non-finite values; "
                            "check its input column for NaN or infinity."
                        )
                    arrays = self._row_arrays(rows, row)
                    row += rows.height
                    yield block, *arrays
            except (_PassAborted, GeneratorExit):
                raise
            except BaseException as exc:
                raise _PassAborted(exc) from exc

        return open_pass

    def _row_arrays(
        self, rows: pl.DataFrame, first_row: int, *, offset: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """y, prior weights and the link-scale offset of a slice, as the in-memory fit
        forms them: log(exposure) [+ offset] [+ the complement's link], weights or ones."""
        model = self.model
        n = rows.height
        y = np.ascontiguousarray(_float64(rows, model.response))
        if model._weights is not None:
            weights = np.ascontiguousarray(model._weights.read(rows, first_row))
        else:
            weights = np.ones(n)
        if not offset:
            return y, weights, None
        link_offset = self._link_offset(rows, first_row)
        if link_offset is None:
            link_offset = np.zeros(n)
        return y, weights, np.ascontiguousarray(link_offset, dtype=np.float64)

    def _link_offset(self, rows: pl.DataFrame, first_row: int) -> np.ndarray | None:
        """The slice's link-scale offset: log(exposure), plus the offset, plus the
        complement on the link scale, in the in-memory fit's order; None without any."""
        model = self.model
        offset = None
        if model._exposure is not None:
            offset = np.log(model._exposure.read(rows, first_row))
        if model._offset is not None:
            user_offset = model._offset.read(rows, first_row)
            offset = user_offset if offset is None else offset + user_offset
        if model._complement is not None:
            complement_link = self._complement_link(rows, first_row)
            offset = complement_link if offset is None else offset + complement_link
        return offset

    def _complement_link(self, rows: pl.DataFrame, first_row: int) -> np.ndarray:
        """A slice's complement on the link scale, as the in-memory fit forms it
        (`_process_complement`): the column or array, or the prior model's prediction
        divided by the raw exposure, checked against the link's domain."""
        from rustystats.formula import GLMModel, apply_link, get_default_link

        model = self.model
        complement = model._complement
        if isinstance(complement, GLMModel):
            values = complement.predict(rows, response_ceiling=None)
            if model._exposure is not None:
                values = values / model._exposure.read(rows, first_row)
        else:
            assert model._complement_input is not None
            values = model._complement_input.read(rows, first_row)
        values = np.asarray(values, dtype=np.float64)
        link = model.link or get_default_link(model.family)
        if link in (None, "log"):
            n_invalid = int(np.sum(values <= 0))
            if n_invalid > 0:
                raise ValidationError(
                    f"Complement values must be strictly positive for {model.family} family "
                    f"with log link. Found {n_invalid} values <= 0."
                )
        elif link == "logit":
            n_invalid = int(np.sum((values <= 0) | (values >= 1)))
            if n_invalid > 0:
                raise ValidationError(
                    f"Complement values must be in (0, 1) for {model.family} family with "
                    f"logit link. Found {n_invalid} values outside range."
                )
        return apply_link(values, link)

    def standardization(
        self, design: FrozenDesign, fit_intercept: bool, open_pass: Callable
    ) -> tuple[np.ndarray, np.ndarray]:
        """Centres and scales from passes M1 and M2 (§6.7), as the in-memory fit
        computes them, then the solver's form (centres zeroed with an intercept)."""
        from rustystats.regularization_path import penalized_column_mask, solver_standardization

        mask = penalized_column_mask(design.p, fit_intercept)
        center, scale, _ridge = compute_standardization_batched_py(
            open_pass,
            self.n_rows,
            design.p,
            weighted=True,
            pen_mask=[bool(v) for v in mask],
            fit_intercept=fit_intercept,
            **self.threads_kwargs,
        )
        return solver_standardization(
            np.asarray(center), np.asarray(scale), fit_intercept=fit_intercept
        )


def _remove_spill(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise SpillError(f"Cannot remove the spill file {path!r}: {exc}") from exc


class _BatchedTargetFolds:
    """The fold designs of an out-of-core fit with target encoding (§7.9): each
    fold's passes encode every row with its design; a fold's spill files go when its
    path ends (``regularization_path.TargetEncodedFolds``)."""

    def __init__(self, run: _Run, designs: list[FrozenDesign], seed: int, theta: float) -> None:
        self.run = run
        self.model = run.model
        self.designs = designs
        self.seed = seed
        self.theta = theta
        #: The fold whose passes are served now, and its ``open_pass`` (one encode
        #: block at a time).
        self._current: tuple[int, Callable[[str], Iterator[tuple]]] | None = None

    def _open(self, fold: int) -> Callable[[str], Iterator[tuple]]:
        if self._current is None or self._current[0] != fold:
            self._current = None
            self._current = (fold, self.run.solver_passes(self.designs[fold]))
        return self._current[1]

    def names(self, fold: int) -> list[str]:
        return list(self.designs[fold].names)

    def moments(self, fold, pen_mask, fit_intercept):
        center, scale, ridge = compute_standardization_batched_py(
            self._open(fold),
            self.run.n_rows,
            self.designs[fold].p,
            weighted=True,
            pen_mask=[bool(v) for v in pen_mask],
            fit_intercept=fit_intercept,
            fold=(fold, len(self.designs), self.seed),
            **self.run.threads_kwargs,
        )
        return np.asarray(center), np.asarray(scale), np.asarray(ridge)

    def alpha_max(
        self, fold, *, family, link, l1_ratio, pen_mask, center, scale, ridge_xtx_diag, trace
    ):
        model = self.model
        return float(
            fold_alpha_max_batched_py(
                self._open(fold),
                self.run.n_rows,
                self.designs[fold].p,
                self.run.y_mean,
                family,
                link,
                float(l1_ratio),
                fold,
                len(self.designs),
                self.seed,
                has_offset=self.run.has_offset,
                var_power=float(model.var_power),
                theta=float(self.theta),
                pen_mask=[bool(v) for v in pen_mask],
                center=center,
                scale=scale,
                ridge_xtx_diag=None
                if ridge_xtx_diag is None
                else [float(v) for v in ridge_xtx_diag],
                allow_extended_tweedie=model.allow_extended_tweedie,
                trace=trace,
                **self.run.kernel_kwargs(self.designs[fold]),
            )
        )

    def path(self, fold, alphas, *, l1_ratio, max_iter, tol, nonneg, nonpos, center, scale):
        model = self.model
        return fold_path_batched_py(
            self._open(fold),
            self.run.n_rows,
            self.designs[fold].p,
            self.run.y_mean,
            model.family,
            fold,
            len(self.designs),
            self.seed,
            link=model.link,
            var_power=model.var_power,
            theta=self.theta,
            has_offset=self.run.has_offset,
            weighted=model._weights_spec is not None,
            alphas=list(alphas),
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            nonneg_indices=nonneg or None,
            nonpos_indices=nonpos or None,
            allow_extended_tweedie=model.allow_extended_tweedie,
            fit_intercept=bool(model.intercept),
            center=center,
            scale=scale,
            **self.run.kernel_kwargs(self.designs[fold]),
        )

    def release(self, fold: int) -> None:
        # Folds run one at a time, so the fold released is the one open, if any.
        self._current = None
        self.run.release_design(self.designs[fold])


def observed_category_labels(
    data: pl.DataFrame | pl.LazyFrame | BatchSource,
    columns: Sequence[str],
    *,
    out_of_core: OutOfCoreOptions | None = None,
) -> dict[str, list[str]]:
    """Sorted labels of each column's non-null values, as ``levels=`` matches them (§4.6).

    Labels use the whole-column conversion and the explicit-level renderer (§6.3), so
    an Int64 column with nulls gives ``"2.0"``. With a :class:`BatchSource` this reads
    only ``columns`` in one ``labels`` pass, bounded and cancellable by
    ``out_of_core``'s ``max_bytes``, ``max_learned_state_bytes`` and ``on_batch``.
    """
    columns = list(columns)
    options = out_of_core if out_of_core is not None else OutOfCoreOptions()
    budget = LearnedStateBudget(options.max_learned_state_bytes)
    if not isinstance(data, BatchSource):
        frame = data.select(columns).collect() if isinstance(data, pl.LazyFrame) else data
        return {
            column: _column_labels(frame[column], column=column, budget=budget)
            for column in columns
        }
    for column in columns:
        dtype = data.schema.get(column)
        if dtype is not None:
            check_key_dtype(dtype, column)
    learn_slice_bytes = _learn_slice_bytes(options.max_bytes)
    working = _ALLOCATOR_ALLOWANCE + 64 * len(columns) + _learning_peak(learn_slice_bytes)
    if options.max_bytes is not None:
        if working > options.max_bytes:
            raise MemoryPlanExceededError(
                f"The labels phase plans a peak of {working:,} bytes, more than "
                f"max_bytes = {options.max_bytes:,}."
            )
        # §6.4: the labels also fit what max_bytes leaves beside the pass's working set.
        budget.cap(
            options.max_bytes - working,
            source=f"max_bytes = {options.max_bytes} (less {working:,} bytes of other state)",
        )
    learners = {column: CategoryKeyLearner(column, budget=budget) for column in columns}

    def admit(column: str, row_bytes: int) -> None:
        # A row larger than a slice is charged before it is learned, as in a fit's
        # learning passes (§4.3).
        budget.admit_row(
            column, row_bytes, transient=_OVERSIZED_ROW_COPIES * row_bytes + _LEARN_ROW_BYTES
        )

    rows_done = 0
    seen = False
    for batch in data.batches(columns):
        for rows in slice_rows(batch, E_LEARN, learn_slice_bytes, admit=admit):
            seen = True
            for column, learner in learners.items():
                learner.observe(rows[column])
            rows_done += rows.height
        if options.on_batch is not None:
            options.on_batch(BatchProgress(0, "labels", rows_done, None))
    if not seen:
        return {column: [] for column in columns}
    return {column: learner._freeze().observed_labels() for column, learner in learners.items()}


def _float64(rows: pl.DataFrame, column: str) -> np.ndarray:
    """A slice's column as float64, as the in-memory fit reads it."""
    return rows[column].to_numpy().astype(np.float64, copy=False)


def _pool_threads() -> int:
    """Threads of the pool a fit without ``threads=`` runs on: Rayon's global pool."""
    try:
        from rustystats._rustystats import rayon_threads_py
    except ImportError:  # pragma: no cover - older builds
        rayon_threads_py = None
    if rayon_threads_py is not None:
        return max(1, int(rayon_threads_py()))
    try:  # pragma: no cover - older builds
        return max(1, int(os.environ.get("RAYON_NUM_THREADS", "")))
    except ValueError:  # pragma: no cover
        return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1)
