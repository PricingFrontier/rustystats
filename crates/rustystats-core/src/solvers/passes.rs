//! Passes over the rows: the one per-row arithmetic every solver driver uses
//! (`specs/out-of-core-fit.md` §5, §7.1, §7.2).
//!
//! A driver never touches rows. It asks a [`PassExecutor`] to run a
//! [`PassRequest`]: a set of *predictors* (how each row's η and μ are derived)
//! and *evaluations* (what to reduce over the rows under those predictors). The
//! executor feeds rows to [`process_rows`] block by block through
//! [`crate::rows`], so the sums are bitwise the same whether the rows are held in
//! memory or arrive in batches, and whatever the thread count.

use std::ops::Range;
use std::sync::Arc;

use ndarray::ArrayView2;

use crate::constants::MAX_IRLS_WEIGHT;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::inference::{leverage_dense_row, leverage_nonzero_row, robust_omega, HCType};
use crate::links::Link;
use crate::regularization::Standardization;
use crate::rows::{reduce_rows, GramPartial, Partial};

use super::initialize_mu_safe_at;
use super::irls::{accumulate_dense_row, accumulate_sparse_slices_pairwise, row_dot};

/// How a row's linear predictor η and mean μ are derived (§7.2).
#[derive(Clone, Debug, PartialEq)]
pub enum Predictor {
    /// η = Xβ + o; μ = clamp(g⁻¹(η)).
    Coef(Arc<[f64]>),
    /// μ = clamp(g⁻¹(Xβ + o)); η = g(μ). A warm start keeps η = g(μ).
    CoefRoundTrip(Arc<[f64]>),
    /// μ from the family initializer (or the safe one); η = g(μ).
    Initializer { safe: bool },
    /// [`Predictor::Initializer`] with the mean response of the rows a fold fits
    /// on, in place of the executor's.
    InitializerAt { safe: bool, y_mean: f64 },
    /// η = η_P + d for a base predictor P; μ = clamp(g⁻¹(η)). The state after an
    /// intercept refresh (`Coef(β)` gives η = (Xβ + o) + d).
    Shifted(Box<Predictor>, f64),
    /// η = Σ_{j∈R} x_j·β_j + o_F over the refit columns R (ascending), with
    /// o_F = ((o + β_{F₁}·x_{F₁}) + β_{F₂}·x_{F₂}) + … over the fixed columns F
    /// (ascending, nonzero coefficients); μ = clamp(g⁻¹(η)). The monotone smooth
    /// refinement's state (`smooth_glm.rs` `refine_unconstrained_block`).
    Partitioned {
        refit: Arc<[usize]>,
        beta: Arc<[f64]>,
        fixed: Arc<[(usize, f64)]>,
    },
}

/// The fixed columns' offset o_F of one row: the offset plus β_c·x_c over the
/// fixed columns in order.
#[inline]
fn fixed_offset(ctx: &RowContext<'_>, x: &RowRef<'_>, fixed: &[(usize, f64)], i: usize) -> f64 {
    let mut offset = ctx.offset[i];
    for &(column, coef) in fixed {
        offset += coef * x.value(column);
    }
    offset
}

/// How α_max's score factor forms dμ/dη and V(μ) (`_glm_score_gradient_factor`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreForm {
    pub link: ScoreLink,
    pub variance: ScoreVariance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreLink {
    Identity,
    Log,
    Logit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreVariance {
    Unit,
    Mu,
    MuSquared,
    Binomial,
    Power(f64),
    NegativeBinomial(f64),
}

/// How a validation fold's μ and unit deviance are formed (`fitting_py.rs`'s
/// validation kernels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationForm {
    /// μ = exp(clamp(η, ±700)) floored at `MU_MIN_POSITIVE`; the Poisson unit
    /// deviance written out.
    PoissonLog,
    /// μ the overflow-safe logistic of η; the family's unit deviance.
    BinomialLogit,
    /// μ = g⁻¹(η); the family's unit deviance.
    Generic,
}

#[inline]
fn score_factor(ctx: &RowContext<'_>, intercept: f64, form: ScoreForm, i: usize) -> f64 {
    let mu = ctx.link.inverse_at(intercept + ctx.offset[i]);
    let dmu_deta = match form.link {
        ScoreLink::Identity => 1.0,
        ScoreLink::Log => mu,
        ScoreLink::Logit => mu * (1.0 - mu),
    };
    let variance = match form.variance {
        ScoreVariance::Unit => 1.0,
        ScoreVariance::Mu => mu,
        ScoreVariance::MuSquared => mu * mu,
        ScoreVariance::Binomial => mu * (1.0 - mu),
        ScoreVariance::Power(power) => mu.powf(power),
        ScoreVariance::NegativeBinomial(theta) => mu + mu * mu / theta,
    };
    let safe_variance = if variance > 0.0 { variance } else { 1.0 };
    ctx.prior_weights[i] * (ctx.y[i] - mu) * dmu_deta / safe_variance
}

#[inline]
fn validation_unit_deviance(ctx: &RowContext<'_>, form: ValidationForm, eta: f64, i: usize) -> f64 {
    let yi = ctx.y[i];
    match form {
        ValidationForm::PoissonLog => {
            let mu = eta
                .clamp(-700.0, 700.0)
                .exp()
                .max(crate::constants::MU_MIN_POSITIVE);
            if yi == 0.0 {
                2.0 * mu
            } else {
                2.0 * (yi * (yi / mu).ln() - (yi - mu))
            }
        }
        ValidationForm::BinomialLogit => {
            let mu = if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                let exp_eta = eta.exp();
                exp_eta / (1.0 + exp_eta)
            };
            ctx.family.unit_deviance_at(yi, mu)
        }
        ValidationForm::Generic => ctx.family.unit_deviance_at(yi, ctx.link.inverse_at(eta)),
    }
}

/// The monotone reparameterisation of a smooth fit's rows (`smooth_glm.rs`
/// `compute_x_tilde_inplace` and `subtract_smooth_offset`), one entry per
/// monotone term.
#[derive(Clone, Debug, PartialEq)]
pub struct MonotoneTransform {
    pub terms: Vec<MonotoneTermTransform>,
}

/// One monotone term: its columns [start, end), the scale of each transformed
/// column (1, then ±exp(α_j)), and its linearisation offset β − Jα.
#[derive(Clone, Debug, PartialEq)]
pub struct MonotoneTermTransform {
    pub start: usize,
    pub end: usize,
    pub scales: Vec<f64>,
    pub offset: Vec<f64>,
}

impl Predictor {
    pub fn coef(beta: &[f64]) -> Self {
        Predictor::Coef(Arc::from(beta))
    }
}

/// Which weights a Gram uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsSpec {
    /// Ordinary IRLS weights at a predictor's μ, times prior weights
    /// (`compute_irls_weights`, `solvers/mod.rs`).
    Irls(usize),
    /// Prior weights.
    Prior,
    /// All zeros (ordinary IRLS `final_weights` before any iteration).
    Zeros,
    /// All ones (smooth `final_weights` before any iteration).
    Ones,
    /// The smooth route's working weights at a predictor, times prior weights
    /// (`smooth_glm.rs` `irls_work_weight`; no true-Hessian weights).
    SmoothIrls(usize),
    /// Prior weights times [`WeightsSpec::SmoothIrls`]: a smooth fit's stored
    /// `irls_weights` times prior weights again, as its robust covariance forms them
    /// (D9).
    PriorTimesSmoothIrls(usize),
}

/// Which response a Gram's right-hand side uses.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseSpec {
    /// The working response z = (η − o) + (y − μ)·g'(μ) at a predictor.
    Working(usize),
    /// η − o at a predictor (the start projection's response).
    LinkMinusOffset(usize),
    /// The working response with the fixed columns' offset o_F in place of o:
    /// (η − o_F) + (y − μ)·g'(μ) (the monotone refinement's).
    PartitionedWorking {
        predictor: usize,
        fixed: Arc<[(usize, f64)]>,
    },
}

/// One quantity reduced over the rows.
#[derive(Clone, Debug, PartialEq)]
pub enum Eval {
    /// Σ w·d(y, μ) and the count of rows whose η or μ is not finite.
    Deviance(usize),
    /// X'WX (packed), X'Wz and z'Wz.
    Gram {
        weights: WeightsSpec,
        response: ResponseSpec,
    },
    /// X'Wz only (no matrix): the same sums a full Gram's right-hand side holds.
    Rhs {
        weights: WeightsSpec,
        response: ResponseSpec,
    },
    /// The count of rows whose family-initializer μ is invalid.
    InitializerInvalid,
    /// The count of rows whose solve weight is positive.
    PositiveWeights(WeightsSpec),
    /// Facts about y, weights and offset alone: whether every prior weight is 1
    /// (within `ZERO_TOL`), the null model's sums (weighted and unweighted), and
    /// max |y| over finite y.
    Facts,
    /// Pearson χ² at a predictor, weighted or not.
    Pearson { predictor: usize, weighted: bool },
    /// The family's log-likelihood sums at a predictor (§8).
    LogLikelihood {
        predictor: usize,
        weighted: bool,
        scale: f64,
    },
    /// The null model's deviance.
    NullDeviance {
        model: crate::diagnostics::NullModel,
        weighted: bool,
    },
    /// Per column: the count of rows whose value is not exactly 1.0.
    NonOneCounts(Arc<[usize]>),
    /// [`Eval::InitializerInvalid`] with a fold's mean response.
    InitializerInvalidAt(f64),
    /// Σy and the row count.
    ResponseSum,
    /// The moment θ₀'s sums at a predictor: Σμ, Σ(y − ȳ)² and the row count
    /// (`diagnostics::estimate_theta_moments`).
    NbMoments { predictor: usize, y_mean: f64 },
    /// The intercept-only fit's closed-form sums: Σw·exp(o), Σw·y, Σw and
    /// Σw·(y − o) (`regularization_path.py` `_fit_null_intercept`).
    NullInterceptSums,
    /// The GLM score at the intercept-only fit β₀ (`_glm_score_gradient_factor`):
    /// Σ x_j·f per column, then Σ f, with f = w(y − μ₀)·dμ/dη / V(μ₀).
    ScoreFactor { intercept: f64, form: ScoreForm },
    /// Σ (x_j·x_j)·w per column.
    WeightedSquares,
    /// A validation fold's deviance sums at coefficients β on the rows as served
    /// (before any row standardization): Σ d (·w) and Σ w (or the row count).
    ValidationDeviance {
        beta: Arc<[f64]>,
        form: ValidationForm,
        weighted: bool,
    },
    /// Per column j: the quasi-score Σ r·x_j and information Σ (f·x_j)·x_j at a
    /// predictor (`smooth_glm.rs` `column_std_score`), as [score, info] pairs.
    Kkt {
        predictor: usize,
        columns: Arc<[usize]>,
    },
    /// Per shift d: the intercept-direction quasi-score Σ w(y − μ(η + d))/(g'V)
    /// at a predictor (`smooth_glm.rs` `intercept_shift_score`).
    ShiftScores {
        predictor: usize,
        shifts: Arc<[f64]>,
    },
    /// X'WX, X'Wz and z'Wz over the columns `columns` only (in their order).
    SubGram {
        columns: Arc<[usize]>,
        weights: WeightsSpec,
        response: ResponseSpec,
    },
    /// X̃'WX̃, X̃'Wz̃ and z̃'Wz̃ on the monotone-transformed rows with the adjusted
    /// working response z̃ = z − Σ_terms x_term·(β − Jα), at a predictor, with the
    /// smooth route's weights (`smooth_glm.rs` monotone PIRLS).
    MonotoneGram {
        predictor: usize,
        transform: Arc<MonotoneTransform>,
    },
    /// One packed robust meat X'ΩX per HC type (`inference::robust_covariance`):
    /// Pearson residuals at `predictor`, combined weights `weights`, leverages from
    /// `bread` (row-major, read when HC2 or HC3 is asked for) and the HC1 factor
    /// `hc1` (`inference::hc1_scale` of the fit's n and p).
    RobustMeats {
        predictor: usize,
        weights: WeightsSpec,
        types: Vec<HCType>,
        bread: Arc<[f64]>,
        hc1: f64,
        /// Form the meats and leverages from the unstandardized rows (with an
        /// original-scale bread), as a stored design would.
        raw_rows: bool,
    },
}

impl WeightsSpec {
    /// This spec with its predictor indices mapped through `map`.
    pub fn remap(self, map: &[usize]) -> Self {
        match self {
            WeightsSpec::Irls(i) => WeightsSpec::Irls(map[i]),
            WeightsSpec::SmoothIrls(i) => WeightsSpec::SmoothIrls(map[i]),
            WeightsSpec::PriorTimesSmoothIrls(i) => WeightsSpec::PriorTimesSmoothIrls(map[i]),
            other => other,
        }
    }
}

impl ResponseSpec {
    /// This spec with its predictor indices mapped through `map`.
    pub fn remap(&self, map: &[usize]) -> Self {
        match self {
            ResponseSpec::Working(i) => ResponseSpec::Working(map[*i]),
            ResponseSpec::LinkMinusOffset(i) => ResponseSpec::LinkMinusOffset(map[*i]),
            ResponseSpec::PartitionedWorking { predictor, fixed } => {
                ResponseSpec::PartitionedWorking {
                    predictor: map[*predictor],
                    fixed: fixed.clone(),
                }
            }
        }
    }
}

impl Eval {
    /// This evaluation with its predictor indices mapped through `map` (old index
    /// → index in a merged request).
    pub fn remap(&self, map: &[usize]) -> Self {
        match self {
            Eval::Deviance(i) => Eval::Deviance(map[*i]),
            Eval::Gram { weights, response } => Eval::Gram {
                weights: weights.remap(map),
                response: response.remap(map),
            },
            Eval::Rhs { weights, response } => Eval::Rhs {
                weights: weights.remap(map),
                response: response.remap(map),
            },
            Eval::SubGram {
                columns,
                weights,
                response,
            } => Eval::SubGram {
                columns: columns.clone(),
                weights: weights.remap(map),
                response: response.remap(map),
            },
            Eval::PositiveWeights(weights) => Eval::PositiveWeights(weights.remap(map)),
            Eval::Pearson {
                predictor,
                weighted,
            } => Eval::Pearson {
                predictor: map[*predictor],
                weighted: *weighted,
            },
            Eval::LogLikelihood {
                predictor,
                weighted,
                scale,
            } => Eval::LogLikelihood {
                predictor: map[*predictor],
                weighted: *weighted,
                scale: *scale,
            },
            Eval::Kkt { predictor, columns } => Eval::Kkt {
                predictor: map[*predictor],
                columns: columns.clone(),
            },
            Eval::ShiftScores { predictor, shifts } => Eval::ShiftScores {
                predictor: map[*predictor],
                shifts: shifts.clone(),
            },
            Eval::NbMoments { predictor, y_mean } => Eval::NbMoments {
                predictor: map[*predictor],
                y_mean: *y_mean,
            },
            Eval::MonotoneGram {
                predictor,
                transform,
            } => Eval::MonotoneGram {
                predictor: map[*predictor],
                transform: transform.clone(),
            },
            Eval::RobustMeats {
                predictor,
                weights,
                types,
                bread,
                hc1,
                raw_rows,
            } => Eval::RobustMeats {
                predictor: map[*predictor],
                weights: weights.remap(map),
                types: types.clone(),
                bread: bread.clone(),
                hc1: *hc1,
                raw_rows: *raw_rows,
            },
            Eval::InitializerInvalid
            | Eval::InitializerInvalidAt(_)
            | Eval::Facts
            | Eval::NullDeviance { .. }
            | Eval::NonOneCounts(_)
            | Eval::ResponseSum
            | Eval::NullInterceptSums
            | Eval::ScoreFactor { .. }
            | Eval::WeightedSquares
            | Eval::ValidationDeviance { .. } => self.clone(),
        }
    }
}

/// What [`Eval::Facts`] reduces.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    /// Rows whose prior weight differs from 1 by `ZERO_TOL` or more.
    pub non_unit_weights: f64,
    /// `diagnostics::null_model_sums` with prior weights.
    pub null_sums_weighted: [f64; 2],
    /// `diagnostics::null_model_sums` without weights.
    pub null_sums_unweighted: [f64; 2],
    /// max |y| over finite y (0 when there is none).
    pub max_abs_y: f64,
}

/// SplitMix64's finaliser.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Which rows of a cross-validation fold an evaluation reads (§7.8): row r is in
/// fold `mix64(seed ^ mix64(r)) mod folds` (DEC-2); training evaluations read
/// the other folds' rows, validation evaluations the fold's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldFilter {
    pub fold: usize,
    pub folds: usize,
    pub seed: u64,
    pub validation: bool,
}

impl FoldFilter {
    /// The fold of global row `row`.
    #[inline]
    pub fn fold_of(row: usize, folds: usize, seed: u64) -> usize {
        (mix64(seed ^ mix64(row as u64)) % folds as u64) as usize
    }

    /// Whether this evaluation reads global row `row`.
    #[inline]
    pub fn includes(&self, row: usize) -> bool {
        (Self::fold_of(row, self.folds, self.seed) == self.fold) == self.validation
    }
}

/// A pass: the predictors its evaluations refer to, and the evaluations, each
/// with the fold filter that selects its rows (None: every row).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassRequest {
    pub predictors: Vec<Predictor>,
    pub evals: Vec<Eval>,
    pub filters: Vec<Option<FoldFilter>>,
    /// The reconstruct pass R (§7.2): the retained state's Gram for the covariance.
    /// Only its name depends on it.
    pub reconstruct: bool,
}

impl PassRequest {
    /// An empty request for the reconstruct pass R (§7.2).
    pub fn reconstruct() -> Self {
        Self {
            reconstruct: true,
            ..Self::default()
        }
    }

    /// Add a predictor and return its index.
    pub fn predictor(&mut self, predictor: Predictor) -> usize {
        if let Some(index) = self.predictors.iter().position(|p| *p == predictor) {
            return index;
        }
        self.predictors.push(predictor);
        self.predictors.len() - 1
    }

    pub fn eval(&mut self, eval: Eval) -> usize {
        self.eval_filtered(eval, None)
    }

    /// Add an evaluation that reads only the rows `filter` selects.
    pub fn eval_filtered(&mut self, eval: Eval, filter: Option<FoldFilter>) -> usize {
        self.evals.push(eval);
        self.filters.push(filter);
        self.evals.len() - 1
    }

    /// What the pass is for, as `BatchProgress.pass_kind` reports it (§4.4): the
    /// start, a candidate (one deviance, with or without a Gram), trials (several
    /// deviances), a Gram, the reconstruct pass, the final statistics, or the
    /// intercept refresh.
    pub fn kind(&self) -> &'static str {
        let has = |test: fn(&Eval) -> bool| self.evals.iter().any(test);
        if self.filters.iter().any(Option::is_some) {
            "fold"
        } else if self.reconstruct {
            "reconstruct"
        } else if has(|e| {
            matches!(
                e,
                Eval::NullInterceptSums | Eval::ScoreFactor { .. } | Eval::WeightedSquares
            )
        }) {
            "null_fit"
        } else if has(|e| matches!(e, Eval::Facts | Eval::InitializerInvalid)) {
            "start"
        } else if has(|e| {
            matches!(
                e,
                Eval::Pearson { .. }
                    | Eval::LogLikelihood { .. }
                    | Eval::NullDeviance { .. }
                    | Eval::Kkt { .. }
                    | Eval::RobustMeats { .. }
            )
        }) {
            "final"
        } else if has(|e| matches!(e, Eval::ShiftScores { .. })) {
            "refresh"
        } else {
            match self
                .evals
                .iter()
                .filter(|e| matches!(e, Eval::Deviance(_)))
                .count()
            {
                0 => "gram",
                1 => "candidate",
                _ => "trials",
            }
        }
    }
}

/// The reduced value of one evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalSum {
    Deviance {
        total: f64,
        nonfinite: f64,
    },
    Gram(GramPartial),
    Rhs(Vec<f64>),
    Count(f64),
    Sum(f64),
    /// Two sums (the Gaussian log-likelihood's squares and total weight).
    Sums2([f64; 2]),
    Facts(Facts),
    /// Packed meats, one per requested HC type (only `xtx_packed` is read).
    Meats(Vec<GramPartial>),
    /// A vector of independent sums.
    Sums(Vec<f64>),
}

impl EvalSum {
    fn zero(eval: &Eval, p: usize) -> Self {
        match eval {
            Eval::Deviance(_) => EvalSum::Deviance {
                total: 0.0,
                nonfinite: 0.0,
            },
            Eval::Gram { .. } => EvalSum::Gram(GramPartial::zeros(p)),
            Eval::Rhs { .. } => EvalSum::Rhs(vec![0.0; p]),
            Eval::InitializerInvalid | Eval::PositiveWeights(_) => EvalSum::Count(0.0),
            Eval::Pearson { .. } | Eval::NullDeviance { .. } => EvalSum::Sum(0.0),
            Eval::LogLikelihood { .. } => EvalSum::Sums2([0.0; 2]),
            Eval::Facts => EvalSum::Facts(Facts {
                non_unit_weights: 0.0,
                null_sums_weighted: [0.0; 2],
                null_sums_unweighted: [0.0; 2],
                max_abs_y: 0.0,
            }),
            Eval::RobustMeats { types, .. } => {
                EvalSum::Meats(types.iter().map(|_| GramPartial::zeros(p)).collect())
            }
            Eval::SubGram { columns, .. } => EvalSum::Gram(GramPartial::zeros(columns.len())),
            Eval::MonotoneGram { .. } => EvalSum::Gram(GramPartial::zeros(p)),
            Eval::NonOneCounts(columns) => EvalSum::Sums(vec![0.0; columns.len()]),
            Eval::InitializerInvalidAt(_) => EvalSum::Count(0.0),
            Eval::ResponseSum => EvalSum::Sums(vec![0.0; 2]),
            Eval::NbMoments { .. } => EvalSum::Sums(vec![0.0; 3]),
            Eval::NullInterceptSums => EvalSum::Sums(vec![0.0; 4]),
            Eval::ScoreFactor { .. } => EvalSum::Sums(vec![0.0; p + 1]),
            Eval::WeightedSquares => EvalSum::Sums(vec![0.0; p]),
            Eval::ValidationDeviance { .. } => EvalSum::Sums(vec![0.0; 2]),
            Eval::Kkt { columns, .. } => EvalSum::Sums(vec![0.0; 2 * columns.len()]),
            Eval::ShiftScores { shifts, .. } => EvalSum::Sums(vec![0.0; shifts.len()]),
        }
    }

    pub fn deviance(&self) -> (f64, bool) {
        match self {
            EvalSum::Deviance { total, nonfinite } => (*total, *nonfinite == 0.0),
            other => panic!("expected a deviance sum, got {other:?}"),
        }
    }

    pub fn into_gram(self) -> GramPartial {
        match self {
            EvalSum::Gram(gram) => gram,
            other => panic!("expected a Gram sum, got {other:?}"),
        }
    }

    pub fn into_rhs(self) -> Vec<f64> {
        match self {
            EvalSum::Rhs(rhs) => rhs,
            other => panic!("expected a right-hand side, got {other:?}"),
        }
    }

    pub fn count(&self) -> f64 {
        match self {
            EvalSum::Count(count) => *count,
            other => panic!("expected a count, got {other:?}"),
        }
    }

    pub fn sum(&self) -> f64 {
        match self {
            EvalSum::Sum(sum) => *sum,
            other => panic!("expected a sum, got {other:?}"),
        }
    }

    pub fn sums2(&self) -> [f64; 2] {
        match self {
            EvalSum::Sums2(sums) => *sums,
            other => panic!("expected two sums, got {other:?}"),
        }
    }

    pub fn facts(&self) -> &Facts {
        match self {
            EvalSum::Facts(facts) => facts,
            other => panic!("expected facts, got {other:?}"),
        }
    }

    pub fn sums(&self) -> &[f64] {
        match self {
            EvalSum::Sums(sums) => sums,
            other => panic!("expected a vector of sums, got {other:?}"),
        }
    }

    pub fn meats(&self) -> &[GramPartial] {
        match self {
            EvalSum::Meats(meats) => meats,
            other => panic!("expected robust meats, got {other:?}"),
        }
    }
}

/// The partial of a whole pass: one sum per evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct PassPartial(pub Vec<EvalSum>);

impl Partial for PassPartial {
    fn reset(&mut self) {
        for sum in &mut self.0 {
            match sum {
                EvalSum::Deviance { total, nonfinite } => {
                    *total = 0.0;
                    *nonfinite = 0.0;
                }
                EvalSum::Gram(gram) => gram.reset(),
                EvalSum::Rhs(rhs) => rhs.reset(),
                EvalSum::Count(count) | EvalSum::Sum(count) => *count = 0.0,
                EvalSum::Sums2(sums) => *sums = [0.0; 2],
                EvalSum::Facts(facts) => {
                    facts.non_unit_weights = 0.0;
                    facts.null_sums_weighted = [0.0; 2];
                    facts.null_sums_unweighted = [0.0; 2];
                    facts.max_abs_y = 0.0;
                }
                EvalSum::Meats(meats) => meats.iter_mut().for_each(GramPartial::reset),
                EvalSum::Sums(sums) => sums.reset(),
            }
        }
    }

    fn add_assign(&mut self, other: &Self) {
        for (sum, add) in self.0.iter_mut().zip(other.0.iter()) {
            match (sum, add) {
                (
                    EvalSum::Deviance { total, nonfinite },
                    EvalSum::Deviance {
                        total: add_total,
                        nonfinite: add_nonfinite,
                    },
                ) => {
                    *total += *add_total;
                    *nonfinite += *add_nonfinite;
                }
                (EvalSum::Gram(gram), EvalSum::Gram(add)) => gram.add_assign(add),
                (EvalSum::Rhs(rhs), EvalSum::Rhs(add)) => rhs.add_assign(add),
                (EvalSum::Count(count), EvalSum::Count(add))
                | (EvalSum::Sum(count), EvalSum::Sum(add)) => *count += *add,
                (EvalSum::Sums2(sums), EvalSum::Sums2(add)) => {
                    sums[0] += add[0];
                    sums[1] += add[1];
                }
                (EvalSum::Facts(facts), EvalSum::Facts(add)) => {
                    facts.non_unit_weights += add.non_unit_weights;
                    for k in 0..2 {
                        facts.null_sums_weighted[k] += add.null_sums_weighted[k];
                        facts.null_sums_unweighted[k] += add.null_sums_unweighted[k];
                    }
                    // A maximum is exact whatever the order.
                    facts.max_abs_y = facts.max_abs_y.max(add.max_abs_y);
                }
                (EvalSum::Meats(meats), EvalSum::Meats(add)) => {
                    for (meat, add) in meats.iter_mut().zip(add.iter()) {
                        meat.add_assign(add);
                    }
                }
                (EvalSum::Sums(sums), EvalSum::Sums(add)) => sums.add_assign(add),
                _ => unreachable!("pass partials share one evaluation layout"),
            }
        }
    }
}

impl PassPartial {
    pub fn zeros(request: &PassRequest, p: usize) -> Self {
        PassPartial(request.evals.iter().map(|e| EvalSum::zero(e, p)).collect())
    }
}

/// What every row computation reads besides X: the family, link and per-row
/// response, offset and prior weights of the rows being processed.
#[derive(Clone, Copy)]
pub struct RowContext<'a> {
    pub family: &'a dyn Family,
    pub link: &'a dyn Link,
    pub y: &'a [f64],
    pub offset: &'a [f64],
    pub prior_weights: &'a [f64],
    /// The unweighted mean response (`families::response_mean` over every row of
    /// the fit), for initializers that read it.
    pub y_mean: f64,
    pub min_weight: f64,
    /// Whether the fit was given an offset (the null model then uses it under a
    /// log-link default family, as `null_deviance_for_family` does).
    pub has_offset: bool,
    /// Columns standardized per row before every kernel (§6.7): the fit runs in
    /// standardized coordinates without a standardized copy of X.
    pub standardization: Option<&'a Standardization>,
}

impl RowContext<'_> {
    fn true_hessian(&self) -> bool {
        self.family.use_true_hessian_weights() && self.link.name() == "log"
    }

    /// Whether the null model scales by exp(offset).
    pub fn null_log_offset(&self) -> bool {
        self.has_offset && self.family.is_log_link_default()
    }
}

/// One row of X, in whichever storage the executor holds.
#[derive(Clone, Copy)]
pub enum RowRef<'a> {
    Dense(&'a [f64]),
    /// The row's nonzeros in ascending column order, with the packed
    /// upper-triangle offset of every column (`irls::packed_upper_offsets`).
    Sparse {
        indices: &'a [u32],
        values: &'a [f64],
        packed_offsets: &'a [usize],
    },
}

impl RowRef<'_> {
    /// The value in column `column` (0.0 where a sparse row holds none).
    #[inline]
    fn value(&self, column: usize) -> f64 {
        match self {
            RowRef::Dense(row) => row[column],
            RowRef::Sparse {
                indices, values, ..
            } => indices
                .binary_search(&(column as u32))
                .map_or(0.0, |position| values[position]),
        }
    }

    #[inline]
    fn dot(&self, beta: &[f64]) -> f64 {
        match self {
            RowRef::Dense(row) => row_dot(row, beta),
            RowRef::Sparse {
                indices, values, ..
            } => {
                // Nonzero products in column order: the dense kernel's bits, since
                // every skipped product is an exact zero.
                let mut sum = 0.0;
                for (&value, &column) in values.iter().zip(indices.iter()) {
                    sum += value * beta[column as usize];
                }
                sum
            }
        }
    }

    #[inline]
    fn accumulate_gram(&self, gram: &mut GramPartial, wk: f64, zk: f64) {
        match self {
            RowRef::Dense(row) => accumulate_dense_row(gram, row, wk, Some(zk)),
            RowRef::Sparse {
                indices,
                values,
                packed_offsets,
            } => {
                let wz = wk * zk;
                gram.ztwz += wz * zk;
                accumulate_sparse_slices_pairwise(
                    indices,
                    values,
                    packed_offsets,
                    &mut gram.xtx_packed,
                    &mut gram.xtz,
                    wk,
                    wz,
                );
            }
        }
    }

    /// Add ω·x·x' to a packed meat: the Gram kernels' (x_j·ω)·x_k products.
    #[inline]
    fn accumulate_meat(&self, meat: &mut GramPartial, omega: f64) {
        match self {
            RowRef::Dense(row) => accumulate_dense_row(meat, row, omega, None),
            RowRef::Sparse {
                indices,
                values,
                packed_offsets,
            } => {
                // The right-hand side receives exact zeros and is never read.
                accumulate_sparse_slices_pairwise(
                    indices,
                    values,
                    packed_offsets,
                    &mut meat.xtx_packed,
                    &mut meat.xtz,
                    omega,
                    0.0,
                );
            }
        }
    }

    /// This row's leverage under `bread` and combined weight `w`.
    #[inline]
    fn leverage(&self, bread: &[f64], w: f64) -> f64 {
        match self {
            RowRef::Dense(row) => leverage_dense_row(row, bread, w),
            RowRef::Sparse {
                indices,
                values,
                packed_offsets,
            } => leverage_nonzero_row(
                indices.len(),
                |a| (indices[a] as usize, values[a]),
                bread,
                packed_offsets.len(),
                w,
            ),
        }
    }

    #[inline]
    fn accumulate_rhs(&self, rhs: &mut [f64], wk: f64, zk: f64) {
        let wz = wk * zk;
        match self {
            RowRef::Dense(row) => {
                for (target, &xki) in rhs.iter_mut().zip(row.iter()) {
                    *target += xki * wz;
                }
            }
            RowRef::Sparse {
                indices, values, ..
            } => {
                for (&value, &column) in values.iter().zip(indices.iter()) {
                    rhs[column as usize] += value * wz;
                }
            }
        }
    }
}

/// Scratch for one standardized row (`RowContext::standardization`).
pub(crate) struct StandardizedRow {
    dense: Vec<f64>,
    indices: Vec<u32>,
    values: Vec<f64>,
    /// Columns with a nonzero centre, ascending: a sparse row gains them.
    centred: Vec<u32>,
}

impl StandardizedRow {
    pub(crate) fn new(standardization: &Standardization) -> Self {
        let p = standardization.center.len();
        Self {
            dense: vec![0.0; p],
            indices: Vec::with_capacity(p),
            values: Vec::with_capacity(p),
            centred: (0..p)
                .filter(|&j| standardization.center[j] != 0.0)
                .map(|j| j as u32)
                .collect(),
        }
    }
}

/// x_j standardized with `Standardization::standardize_matrix`'s expression: unchanged
/// when the column has centre 0 and scale 1, else (x_j − c_j)/s_j.
#[inline]
fn standardized_value(standardization: &Standardization, j: usize, x: f64) -> f64 {
    let (center, scale) = (standardization.center[j], standardization.scale[j]);
    if center == 0.0 && scale == 1.0 {
        x
    } else {
        (x - center) / scale
    }
}

/// `raw` standardized into `scratch`. A dense row stays dense; a sparse row stays
/// sparse, gaining the centred columns. Each value is the one
/// `Standardization::standardize_matrix` writes, so every kernel gives the bits it
/// gives on the standardized matrix.
#[inline]
pub(crate) fn standardize_row<'b>(
    raw: RowRef<'b>,
    standardization: &Standardization,
    scratch: &'b mut StandardizedRow,
) -> RowRef<'b> {
    match raw {
        RowRef::Dense(row) => {
            for (j, (target, &x)) in scratch.dense.iter_mut().zip(row.iter()).enumerate() {
                *target = standardized_value(standardization, j, x);
            }
            RowRef::Dense(&scratch.dense)
        }
        RowRef::Sparse {
            indices,
            values,
            packed_offsets,
        } => {
            scratch.indices.clear();
            scratch.values.clear();
            let (mut a, mut b) = (0, 0);
            while a < indices.len() || b < scratch.centred.len() {
                let next_raw = indices.get(a).copied().unwrap_or(u32::MAX);
                let next_centred = scratch.centred.get(b).copied().unwrap_or(u32::MAX);
                let (column, x) = if next_raw <= next_centred {
                    a += 1;
                    if next_raw == next_centred {
                        b += 1;
                    }
                    (next_raw, values[a - 1])
                } else {
                    b += 1;
                    (next_centred, 0.0)
                };
                scratch.indices.push(column);
                scratch
                    .values
                    .push(standardized_value(standardization, column as usize, x));
            }
            RowRef::Sparse {
                indices: &scratch.indices,
                values: &scratch.values,
                packed_offsets,
            }
        }
    }
}

/// η and μ of one row under one predictor.
#[derive(Clone, Copy, Debug, Default)]
struct RowState {
    eta: f64,
    mu: f64,
}

#[inline]
fn initializer_state(ctx: &RowContext<'_>, safe: bool, y_mean: f64, i: usize) -> RowState {
    let mu = if safe {
        initialize_mu_safe_at(ctx.family, ctx.y[i], y_mean)
    } else {
        ctx.family.initialize_mu_at(ctx.y[i], y_mean)
    };
    RowState {
        eta: ctx.link.link_at(mu),
        mu,
    }
}

#[inline]
fn row_state(predictor: &Predictor, ctx: &RowContext<'_>, x: &RowRef<'_>, i: usize) -> RowState {
    let family = ctx.family;
    let link = ctx.link;
    match predictor {
        Predictor::Coef(beta) => {
            let eta = x.dot(beta) + ctx.offset[i];
            RowState {
                eta,
                mu: family.clamp_mu_at(link.inverse_at(eta)),
            }
        }
        Predictor::CoefRoundTrip(beta) => {
            let mu = family.clamp_mu_at(link.inverse_at(x.dot(beta) + ctx.offset[i]));
            RowState {
                eta: link.link_at(mu),
                mu,
            }
        }
        Predictor::Initializer { safe } => initializer_state(ctx, *safe, ctx.y_mean, i),
        Predictor::InitializerAt { safe, y_mean } => initializer_state(ctx, *safe, *y_mean, i),
        Predictor::Partitioned { refit, beta, fixed } => {
            let mut dot = 0.0;
            for (&column, &coef) in refit.iter().zip(beta.iter()) {
                dot += x.value(column) * coef;
            }
            let eta = dot + fixed_offset(ctx, x, fixed, i);
            RowState {
                eta,
                mu: family.clamp_mu_at(link.inverse_at(eta)),
            }
        }
        Predictor::Shifted(base, shift) => {
            let eta = row_state(base, ctx, x, i).eta + shift;
            RowState {
                eta,
                mu: family.clamp_mu_at(link.inverse_at(eta)),
            }
        }
    }
}

/// The ordinary IRLS weight of one row (without prior weights), as
/// `compute_irls_weights` forms it.
#[inline]
fn irls_weight(ctx: &RowContext<'_>, state: RowState, i: usize) -> f64 {
    if ctx.true_hessian() {
        ctx.family
            .true_hessian_weight_at(state.mu, ctx.y[i])
            .max(ctx.min_weight)
            .min(MAX_IRLS_WEIGHT)
    } else {
        let d = ctx.link.derivative_at(state.mu);
        (1.0 / (ctx.family.variance_at(state.mu) * d * d))
            .max(ctx.min_weight)
            .min(MAX_IRLS_WEIGHT)
    }
}

/// The smooth route's working weight of one row (without prior weights).
#[inline]
fn smooth_irls_weight(ctx: &RowContext<'_>, state: RowState) -> f64 {
    let d = ctx.link.derivative_at(state.mu);
    (1.0 / (ctx.family.variance_at(state.mu) * d * d))
        .max(ctx.min_weight)
        .min(MAX_IRLS_WEIGHT)
}

#[inline]
fn solve_weight(ctx: &RowContext<'_>, spec: WeightsSpec, states: &[RowState], i: usize) -> f64 {
    match spec {
        WeightsSpec::Irls(index) => ctx.prior_weights[i] * irls_weight(ctx, states[index], i),
        WeightsSpec::Prior => ctx.prior_weights[i],
        WeightsSpec::Zeros => 0.0,
        WeightsSpec::Ones => 1.0,
        WeightsSpec::SmoothIrls(index) => {
            ctx.prior_weights[i] * smooth_irls_weight(ctx, states[index])
        }
        WeightsSpec::PriorTimesSmoothIrls(index) => {
            ctx.prior_weights[i] * (ctx.prior_weights[i] * smooth_irls_weight(ctx, states[index]))
        }
    }
}

/// `guarded_denominator` of `smooth_glm.rs`: a quasi-score denominator kept away
/// from zero with its sign.
#[inline]
fn guarded_denominator(denom: f64) -> f64 {
    if denom.abs() < crate::constants::ZERO_TOL {
        crate::constants::ZERO_TOL.copysign(denom)
    } else {
        denom
    }
}

/// The quasi-score residual w(y − μ)/(g'(μ)V(μ)) and expected-information weight
/// w/(g'(μ)²V(μ)) of one row (`smooth_glm.rs` `quasi_score_arrays`).
#[inline]
fn quasi_score_terms(ctx: &RowContext<'_>, mu: f64, i: usize) -> (f64, f64) {
    let link_deriv = ctx.link.derivative_at(mu);
    let variance = ctx.family.variance_at(mu);
    let w = ctx.prior_weights[i];
    let denom = guarded_denominator(link_deriv * variance);
    let resid = w * (ctx.y[i] - mu) / denom;
    let info_denom = (link_deriv * link_deriv * variance).max(crate::constants::ZERO_TOL);
    (resid, w / info_denom)
}

#[inline]
fn response(
    ctx: &RowContext<'_>,
    spec: &ResponseSpec,
    states: &[RowState],
    x: &RowRef<'_>,
    i: usize,
) -> f64 {
    match spec {
        ResponseSpec::Working(index) => {
            let state = states[*index];
            let d = ctx.link.derivative_at(state.mu);
            (state.eta - ctx.offset[i]) + (ctx.y[i] - state.mu) * d
        }
        ResponseSpec::LinkMinusOffset(index) => states[*index].eta - ctx.offset[i],
        ResponseSpec::PartitionedWorking { predictor, fixed } => {
            let state = states[*predictor];
            let d = ctx.link.derivative_at(state.mu);
            (state.eta - fixed_offset(ctx, x, fixed, i)) + (ctx.y[i] - state.mu) * d
        }
    }
}

/// `transform`'s row: each monotone term's columns replaced by
/// scale_j · Σ_{l ≥ j} x_l (reverse cumulative sums within the term), written
/// into `buffer`; and the working response `z` adjusted by each term's
/// Σ x_l·offset_l, in term order.
#[inline]
fn monotone_row(
    x: &RowRef<'_>,
    transform: &MonotoneTransform,
    z: f64,
    buffer: &mut Vec<f64>,
    p: usize,
) -> f64 {
    buffer.clear();
    buffer.resize(p, 0.0);
    match x {
        RowRef::Dense(row) => buffer.copy_from_slice(row),
        RowRef::Sparse {
            indices, values, ..
        } => {
            for (&column, &value) in indices.iter().zip(values.iter()) {
                buffer[column as usize] = value;
            }
        }
    }
    let mut adjusted = z;
    for term in &transform.terms {
        let mut dot = 0.0;
        for (column, &offset) in (term.start..term.end).zip(term.offset.iter()) {
            dot += buffer[column] * offset;
        }
        adjusted -= dot;
    }
    for term in &transform.terms {
        let mut cumsum = 0.0;
        for column in (term.start..term.end).rev() {
            cumsum += buffer[column];
            buffer[column] = term.scales[column - term.start] * cumsum;
        }
    }
    adjusted
}

/// η and μ under `predictor` of the rows `rows`, written to `eta[i - rows.start]`
/// and `mu[i - rows.start]`: the state every evaluation reads, row standardization
/// included.
pub fn row_states<'r, F>(
    predictor: &Predictor,
    ctx: &RowContext<'_>,
    rows: Range<usize>,
    eta: &mut [f64],
    mu: &mut [f64],
    row: F,
) where
    F: Fn(usize) -> RowRef<'r>,
{
    let mut scratch = ctx.standardization.map(StandardizedRow::new);
    let start = rows.start;
    for i in rows {
        let raw = row(i);
        let x = match (ctx.standardization, scratch.as_mut()) {
            (Some(standardization), Some(scratch)) => {
                standardize_row(raw, standardization, scratch)
            }
            _ => raw,
        };
        let state = row_state(predictor, ctx, &x, i);
        eta[i - start] = state.eta;
        mu[i - start] = state.mu;
    }
}

/// Add the rows `rows` to `partial`. `row(i)` returns local row `i` of X; `i`
/// also indexes `ctx`'s per-row slices.
pub fn process_rows<'r, F>(
    request: &PassRequest,
    ctx: &RowContext<'_>,
    rows: Range<usize>,
    first_row: usize,
    partial: &mut PassPartial,
    row: F,
) where
    F: Fn(usize) -> RowRef<'r>,
{
    let mut states = vec![RowState::default(); request.predictors.len()];
    let mut scratch = ctx.standardization.map(StandardizedRow::new);
    let mut buffer: Vec<f64> = Vec::new();
    for i in rows {
        let raw = row(i);
        let x = match (ctx.standardization, scratch.as_mut()) {
            (Some(standardization), Some(scratch)) => {
                standardize_row(raw, standardization, scratch)
            }
            _ => raw,
        };
        for (state, predictor) in states.iter_mut().zip(request.predictors.iter()) {
            *state = row_state(predictor, ctx, &x, i);
        }
        let global_row = first_row + i;
        for (index, (eval, sum)) in request.evals.iter().zip(partial.0.iter_mut()).enumerate() {
            if let Some(Some(filter)) = request.filters.get(index) {
                if !filter.includes(global_row) {
                    continue;
                }
            }
            match (eval, sum) {
                (Eval::Deviance(index), EvalSum::Deviance { total, nonfinite }) => {
                    let state = states[*index];
                    if !(state.eta.is_finite() && state.mu.is_finite()) {
                        *nonfinite += 1.0;
                    }
                    *total +=
                        ctx.family.unit_deviance_at(ctx.y[i], state.mu) * ctx.prior_weights[i];
                }
                (
                    Eval::Gram {
                        weights,
                        response: resp,
                    },
                    EvalSum::Gram(gram),
                ) => {
                    let wk = solve_weight(ctx, *weights, &states, i);
                    let zk = response(ctx, resp, &states, &x, i);
                    x.accumulate_gram(gram, wk, zk);
                }
                (
                    Eval::Rhs {
                        weights,
                        response: resp,
                    },
                    EvalSum::Rhs(rhs),
                ) => {
                    let wk = solve_weight(ctx, *weights, &states, i);
                    let zk = response(ctx, resp, &states, &x, i);
                    x.accumulate_rhs(rhs, wk, zk);
                }
                (Eval::PositiveWeights(weights), EvalSum::Count(count)) => {
                    if solve_weight(ctx, *weights, &states, i) > 0.0 {
                        *count += 1.0;
                    }
                }
                (Eval::Facts, EvalSum::Facts(facts)) => {
                    let (yi, wi) = (ctx.y[i], ctx.prior_weights[i]);
                    if (wi - 1.0).abs() >= crate::constants::ZERO_TOL {
                        facts.non_unit_weights += 1.0;
                    }
                    if ctx.null_log_offset() {
                        let oi = ctx.offset[i];
                        facts.null_sums_weighted[0] += oi.exp() * wi;
                        facts.null_sums_weighted[1] += yi * wi;
                        facts.null_sums_unweighted[0] += oi.exp();
                        facts.null_sums_unweighted[1] += yi;
                    } else {
                        facts.null_sums_weighted[0] += yi * wi;
                        facts.null_sums_weighted[1] += wi;
                        facts.null_sums_unweighted[0] += yi;
                    }
                    if yi.is_finite() {
                        facts.max_abs_y = facts.max_abs_y.max(yi.abs());
                    }
                }
                (
                    Eval::Pearson {
                        predictor,
                        weighted,
                    },
                    EvalSum::Sum(total),
                ) => {
                    let term = crate::diagnostics::pearson_row(
                        ctx.family,
                        ctx.y[i],
                        states[*predictor].mu,
                    );
                    *total += if *weighted {
                        term * ctx.prior_weights[i]
                    } else {
                        term
                    };
                }
                (
                    Eval::LogLikelihood {
                        predictor,
                        weighted,
                        scale,
                    },
                    EvalSum::Sums2(sums),
                ) => {
                    let (yi, mu) = (ctx.y[i], states[*predictor].mu);
                    let wi = ctx.prior_weights[i];
                    match ctx.family.log_likelihood_form() {
                        crate::families::LogLikelihoodForm::Gaussian => {
                            let diff = yi - mu;
                            if *weighted {
                                sums[0] += wi * diff * diff;
                                sums[1] += wi;
                            } else {
                                sums[0] += diff * diff;
                            }
                        }
                        crate::families::LogLikelihoodForm::RowSum { .. } => {
                            let term = ctx.family.log_likelihood_row(yi, mu, *scale);
                            sums[0] += if *weighted { term * wi } else { term };
                        }
                        crate::families::LogLikelihoodForm::DevianceFallback => {
                            sums[0] += ctx.family.unit_deviance_at(yi, mu);
                        }
                    }
                }
                (Eval::NullDeviance { model, weighted }, EvalSum::Sum(total)) => {
                    let offset = ctx.null_log_offset().then(|| ctx.offset[i]);
                    let unit = ctx.family.unit_deviance_at(ctx.y[i], model.mean_at(offset));
                    *total += if *weighted {
                        unit * ctx.prior_weights[i]
                    } else {
                        unit
                    };
                }
                (
                    Eval::RobustMeats {
                        predictor,
                        weights,
                        types,
                        bread,
                        hc1,
                        raw_rows,
                    },
                    EvalSum::Meats(meats),
                ) => {
                    let rows_for_meat = if *raw_rows { raw } else { x };
                    // As `inference::robust_covariance`: combined weight prior × IRLS,
                    // Pearson residual at the fitted μ, leverage for HC2/HC3 only.
                    let w = solve_weight(ctx, *weights, &states, i);
                    let r = crate::diagnostics::pearson_residual_at(
                        ctx.family,
                        ctx.y[i],
                        states[*predictor].mu,
                    );
                    let h = if types.iter().any(|t| matches!(t, HCType::HC2 | HCType::HC3)) {
                        rows_for_meat.leverage(bread, w)
                    } else {
                        0.0
                    };
                    for (hc_type, meat) in types.iter().zip(meats.iter_mut()) {
                        rows_for_meat.accumulate_meat(meat, robust_omega(*hc_type, r, w, h, *hc1));
                    }
                }
                (
                    Eval::SubGram {
                        columns,
                        weights,
                        response: resp,
                    },
                    EvalSum::Gram(gram),
                ) => {
                    let wk = solve_weight(ctx, *weights, &states, i);
                    let zk = response(ctx, resp, &states, &x, i);
                    buffer.clear();
                    buffer.extend(columns.iter().map(|&column| x.value(column)));
                    accumulate_dense_row(gram, &buffer, wk, Some(zk));
                }
                (
                    Eval::MonotoneGram {
                        predictor,
                        transform,
                    },
                    EvalSum::Gram(gram),
                ) => {
                    let wk = solve_weight(ctx, WeightsSpec::SmoothIrls(*predictor), &states, i);
                    let z = response(ctx, &ResponseSpec::Working(*predictor), &states, &x, i);
                    let p = gram.p();
                    let zk = monotone_row(&x, transform, z, &mut buffer, p);
                    accumulate_dense_row(gram, &buffer, wk, Some(zk));
                }
                (Eval::InitializerInvalidAt(y_mean), EvalSum::Count(count)) => {
                    let mu = ctx.family.initialize_mu_at(ctx.y[i], *y_mean);
                    if !ctx.family.is_valid_mu_at(mu) {
                        *count += 1.0;
                    }
                }
                (Eval::NbMoments { predictor, y_mean }, EvalSum::Sums(sums)) => {
                    sums[0] += states[*predictor].mu;
                    sums[1] += (ctx.y[i] - y_mean).powi(2);
                    sums[2] += 1.0;
                }
                (Eval::ResponseSum, EvalSum::Sums(sums)) => {
                    sums[0] += ctx.y[i];
                    sums[1] += 1.0;
                }
                (Eval::NullInterceptSums, EvalSum::Sums(sums)) => {
                    let (yi, wi, oi) = (ctx.y[i], ctx.prior_weights[i], ctx.offset[i]);
                    sums[0] += wi * oi.exp();
                    sums[1] += wi * yi;
                    sums[2] += wi;
                    sums[3] += wi * (yi - oi);
                }
                (Eval::ScoreFactor { intercept, form }, EvalSum::Sums(sums)) => {
                    let f = score_factor(ctx, *intercept, *form, i);
                    let p = sums.len() - 1;
                    for (j, target) in sums[..p].iter_mut().enumerate() {
                        *target += x.value(j) * f;
                    }
                    sums[p] += f;
                }
                (Eval::WeightedSquares, EvalSum::Sums(sums)) => {
                    let wi = ctx.prior_weights[i];
                    for (j, target) in sums.iter_mut().enumerate() {
                        let v = x.value(j);
                        *target += v * v * wi;
                    }
                }
                (
                    Eval::ValidationDeviance {
                        beta,
                        form,
                        weighted,
                    },
                    EvalSum::Sums(sums),
                ) => {
                    let eta = raw.dot(beta) + ctx.offset[i];
                    let unit = validation_unit_deviance(ctx, *form, eta, i);
                    if *weighted {
                        let wi = ctx.prior_weights[i];
                        sums[0] += unit * wi;
                        sums[1] += wi;
                    } else {
                        sums[0] += unit;
                        sums[1] += 1.0;
                    }
                }
                (Eval::NonOneCounts(columns), EvalSum::Sums(counts)) => {
                    for (count, &column) in counts.iter_mut().zip(columns.iter()) {
                        if x.value(column) != 1.0 {
                            *count += 1.0;
                        }
                    }
                }
                (Eval::Kkt { predictor, columns }, EvalSum::Sums(sums)) => {
                    let (resid, info_weight) = quasi_score_terms(ctx, states[*predictor].mu, i);
                    for (pair, &column) in
                        sums.as_chunks_mut::<2>().0.iter_mut().zip(columns.iter())
                    {
                        let v = x.value(column);
                        pair[0] += resid * v;
                        pair[1] += info_weight * v * v;
                    }
                }
                (Eval::ShiftScores { predictor, shifts }, EvalSum::Sums(sums)) => {
                    let eta = states[*predictor].eta;
                    for (sum, &shift) in sums.iter_mut().zip(shifts.iter()) {
                        let mu = ctx.family.clamp_mu_at(ctx.link.inverse_at(eta + shift));
                        let denom = guarded_denominator(
                            ctx.link.derivative_at(mu) * ctx.family.variance_at(mu),
                        );
                        *sum += ctx.prior_weights[i] * (ctx.y[i] - mu) / denom;
                    }
                }
                (Eval::InitializerInvalid, EvalSum::Count(count)) => {
                    let mu = ctx.family.initialize_mu_at(ctx.y[i], ctx.y_mean);
                    if !ctx.family.is_valid_mu_at(mu) {
                        *count += 1.0;
                    }
                }
                _ => unreachable!("partial layout follows the request"),
            }
        }
    }
}

fn unimplemented_capability(name: &str) {
    panic!("this executor does not support {name}");
}

// ── Row kernels (§5.2) ───────────────────────────────────────────────────────

/// A design whose rows hold at most this share of nonzero columns is served to the
/// kernels as sparse rows.
pub const SPARSE_ROW_DENSITY_THRESHOLD: f64 = 0.35;
/// Sparse rows only pay off on designs of at least this many cells (n·p).
pub const SPARSE_ROW_MIN_CELLS: usize = 10_000_000;

/// Whether the executors serve rows as sparse rows (§5.2): from the design's
/// structural maximum nonzeros per row, p and n only, never sampled values or the
/// thread count. Both kernels give the same bits (a skipped product is an exact
/// zero), so this only affects speed.
pub fn sparse_rows_for(n: usize, p: usize, max_row_nonzeros: usize) -> bool {
    if n == 0 || p < 16 {
        return false;
    }
    (max_row_nonzeros as f64) / (p as f64) <= SPARSE_ROW_DENSITY_THRESHOLD
        && n.saturating_mul(p) >= SPARSE_ROW_MIN_CELLS
}

/// The largest count of nonzeros in any row of a C-order design: an in-memory
/// design's own bound when the caller gives no structural one.
pub fn design_max_row_nonzeros(x: &[f64], p: usize) -> usize {
    if p == 0 {
        return 0;
    }
    x.chunks_exact(p)
        .map(|row| row.iter().filter(|&&value| value != 0.0).count())
        .max()
        .unwrap_or(0)
}

/// Which row kernel a fit's executor serves (§5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKernel {
    /// Chosen by [`sparse_rows_for`] from n, p and the design's structural bound on
    /// nonzeros per row (None: no bound is given).
    Structural(Option<usize>),
    /// Dense rows whatever the design (the explicit choice of a test or benchmark).
    Dense,
    /// Sparse rows whatever the design (the explicit choice of a test or benchmark).
    Sparse,
}

impl Default for RowKernel {
    fn default() -> Self {
        RowKernel::Structural(None)
    }
}

impl RowKernel {
    /// Whether an out-of-core executor over `n` rows of `p` columns serves sparse
    /// rows; with no structural bound it serves dense rows.
    pub fn sparse_out_of_core(self, n: usize, p: usize) -> bool {
        match self {
            RowKernel::Structural(bound) => bound.is_some_and(|k| sparse_rows_for(n, p, k)),
            RowKernel::Dense => false,
            RowKernel::Sparse => true,
        }
    }

    /// Whether an in-memory executor over `x` serves sparse rows. With no structural
    /// bound, `x`'s own rows give it, scanned only when n and p allow sparse rows.
    pub fn sparse_in_memory(self, x: ArrayView2<'_, f64>) -> bool {
        let (n, p) = x.dim();
        match self {
            RowKernel::Structural(_) if !sparse_rows_for(n, p, 0) => false,
            RowKernel::Structural(Some(bound)) => sparse_rows_for(n, p, bound),
            RowKernel::Structural(None) => x
                .as_slice()
                .is_some_and(|slice| sparse_rows_for(n, p, design_max_row_nonzeros(slice, p))),
            RowKernel::Dense => false,
            RowKernel::Sparse => true,
        }
    }
}

/// The packed upper-triangle offsets an out-of-core executor needs when `kernel`
/// serves sparse rows for `n` rows of `p` columns; None for dense rows.
pub fn sparse_row_offsets(n: usize, p: usize, kernel: RowKernel) -> Option<Vec<usize>> {
    kernel
        .sparse_out_of_core(n, p)
        .then(|| crate::solvers::irls::packed_upper_offsets(p))
}

/// Whether a fit whose rows are standardized by `standardization` keeps sparse rows:
/// a centred column makes every row dense in it, so only a scale-only (or no)
/// standardization does.
pub fn rows_stay_sparse(standardization: Option<&Standardization>) -> bool {
    standardization.is_none_or(|s| s.center.iter().all(|&c| c == 0.0))
}

/// The rows of one block of a C-order design as compressed sparse rows, built for a
/// block's reduction and dropped after it: at most `GRAM_BLOCK_ROWS` rows, so its
/// size is bounded by the block and the design's nonzeros per row (§11).
pub(crate) struct BlockCsr {
    first: usize,
    offsets: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl BlockCsr {
    /// Rows `rows` of `x` (C-order, `p` columns per row). The vectors are allocated
    /// once at their exact lengths (the block's nonzeros, counted first), so the
    /// CSR holds 12 bytes per nonzero plus 8 per row, never more (§11).
    pub(crate) fn build(x: &[f64], p: usize, rows: Range<usize>) -> Self {
        let block = &x[rows.start * p..rows.end * p];
        let nonzeros = block.iter().filter(|&&value| value != 0.0).count();
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut indices = Vec::with_capacity(nonzeros);
        let mut values = Vec::with_capacity(nonzeros);
        offsets.push(0);
        for row in rows.clone() {
            for (column, &value) in x[row * p..(row + 1) * p].iter().enumerate() {
                if value != 0.0 {
                    indices.push(column as u32);
                    values.push(value);
                }
            }
            offsets.push(indices.len());
        }
        BlockCsr {
            first: rows.start,
            offsets,
            indices,
            values,
        }
    }

    /// Row `row` (an index into `x`), its nonzeros in ascending column order.
    #[inline]
    pub(crate) fn row<'s>(&'s self, row: usize, packed_offsets: &'s [usize]) -> RowRef<'s> {
        let local = row - self.first;
        let (start, end) = (self.offsets[local], self.offsets[local + 1]);
        RowRef::Sparse {
            indices: &self.indices[start..end],
            values: &self.values[start..end],
            packed_offsets,
        }
    }
}

/// Process rows `rows` of a dense C-order block `x` (local row `i` is global row
/// `first_row + i`), served dense, as sparse rows when `packed_offsets` is given
/// (§5.2), or as the single column [1.0] for an intercept-only fit.
#[allow(clippy::too_many_arguments)]
pub fn process_block(
    request: &PassRequest,
    ctx: &RowContext<'_>,
    x: &[f64],
    p: usize,
    rows: Range<usize>,
    first_row: usize,
    partial: &mut PassPartial,
    packed_offsets: Option<&[usize]>,
    intercept_only: bool,
) {
    if intercept_only {
        process_rows(request, ctx, rows, first_row, partial, |_| {
            RowRef::Dense(&ONE_COLUMN)
        });
        return;
    }
    match packed_offsets {
        Some(packed) => {
            let csr = BlockCsr::build(x, p, rows.clone());
            process_rows(request, ctx, rows, first_row, partial, |i| {
                csr.row(i, packed)
            });
        }
        None => process_rows(request, ctx, rows, first_row, partial, |i| {
            RowRef::Dense(&x[i * p..(i + 1) * p])
        }),
    }
}

/// One semantic step of a solver driver (`specs/out-of-core-fit.md` §4.5).
///
/// Drivers record a step at each of their decisions, never per pass, so the trace
/// is the same whatever the executor speculates, batches or spills.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverStep {
    /// The driver: `irls`, `cd`, `smooth`, `monotone`, `alpha_max`, `cv` or `nb_theta`.
    pub driver: &'static str,
    /// What the step decided (§4.5's table).
    pub kind: &'static str,
    /// Semantic evaluations the decision read, in the driver's sequential order.
    pub evaluations: usize,
    pub accepted: bool,
    pub status: String,
}

impl SolverStep {
    pub fn new(
        driver: &'static str,
        kind: &'static str,
        evaluations: usize,
        accepted: bool,
        status: impl Into<String>,
    ) -> Self {
        SolverStep {
            driver,
            kind,
            evaluations,
            accepted,
            status: status.into(),
        }
    }
}

/// Record `step` in the executor's trace (an executor that keeps none drops it).
pub fn record_step(executor: &mut dyn PassExecutor, step: SolverStep) {
    if let Some(trace) = executor.trace() {
        trace.push(step);
    }
}

/// The length of the executor's trace (0 when it keeps none).
pub fn trace_len(executor: &mut dyn PassExecutor) -> usize {
    executor.trace().map_or(0, |trace| trace.len())
}

/// Drop the steps recorded after the trace had `len` steps: a driver nested inside
/// another (alpha_max's intercept-only IRLS) is summarized by the outer one.
pub fn truncate_trace(executor: &mut dyn PassExecutor, len: usize) {
    if let Some(trace) = executor.trace() {
        trace.truncate(len);
    }
}

/// Serves passes: in memory from X, or out of core from batches.
pub trait PassExecutor {
    fn n_rows(&self) -> usize;
    fn n_params(&self) -> usize;
    fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>>;
    /// Standardize every row before the kernels from now on (§6.7), or stop.
    fn set_row_standardization(&mut self, standardization: Option<Standardization>);
    /// Serve every row as the single column [1.0] from now on (an intercept-only
    /// fit over the same rows), or stop.
    fn set_intercept_only(&mut self, intercept_only: bool);
    /// Evaluate rows with `family` from now on (Negative Binomial θ rounds).
    fn set_family(&mut self, _family: Box<dyn Family>) {
        unimplemented_capability("set_family");
    }
    /// Keep μ of every row under `predictor` for later θ passes (§7.10): in RAM in
    /// memory, in a spill file out of core.
    fn spill_mu(&mut self, _predictor: &Predictor) -> Result<()> {
        Err(RustyStatsError::InvalidValue(
            "this executor cannot keep μ for θ estimation".to_string(),
        ))
    }
    /// Σ w·ℓ(y, μ; θ) (or Σ ℓ) over the spilled μ, block-ordered
    /// (`diagnostics::nb_loglikelihood`).
    fn spilled_nb_loglik(&mut self, _theta: f64, _weighted: bool) -> Result<f64> {
        Err(RustyStatsError::InvalidValue(
            "this executor cannot evaluate θ passes".to_string(),
        ))
    }
    /// Whether a pass costs much more than its row arithmetic (rows read from a
    /// source), so drivers should evaluate more per pass to take fewer passes.
    /// Speculation never changes results.
    fn prefers_few_passes(&self) -> bool {
        false
    }
    /// The solver trace the drivers record into (§4.5), when this executor keeps
    /// one; a wrapper forwards its inner executor's.
    fn trace(&mut self) -> Option<&mut Vec<SolverStep>> {
        None
    }
}

/// Serves passes from a design held in memory.
pub struct InMemoryExecutor<'a> {
    x: ArrayView2<'a, f64>,
    /// The packed upper-triangle offsets of every column when rows are served as
    /// sparse rows, block by block (§5.2); None for dense rows.
    sparse_offsets: Option<Vec<usize>>,
    /// The per-row context; its `standardization` is replaced by the executor's own.
    pub ctx: RowContext<'a>,
    standardization: Option<Standardization>,
    /// Serve every row as the single column [1.0] (`set_intercept_only`).
    intercept_only: bool,
    /// The family rows are evaluated with, in place of `ctx.family` (`set_family`).
    family_override: Option<Box<dyn Family>>,
    /// μ of every row, kept by `spill_mu`.
    spilled_mu: Option<Vec<f64>>,
    /// The drivers' steps (§4.5).
    trace: Vec<SolverStep>,
}

impl<'a> InMemoryExecutor<'a> {
    /// An executor over `x`, whose passes serve its rows dense or, when
    /// `sparse_rows`, as sparse rows formed per block (§5.2; [`sparse_rows_for`]).
    pub fn new(x: ArrayView2<'a, f64>, sparse_rows: bool, ctx: RowContext<'a>) -> Result<Self> {
        if x.as_slice().is_none() {
            return Err(RustyStatsError::LinearAlgebraError(
                "Design matrix X must be contiguous in memory (C-order)".to_string(),
            ));
        }
        Ok(Self {
            sparse_offsets: sparse_rows
                .then(|| crate::solvers::irls::packed_upper_offsets(x.ncols())),
            x,
            ctx,
            standardization: None,
            intercept_only: false,
            family_override: None,
            spilled_mu: None,
            trace: Vec::new(),
        })
    }

    /// The per-row context with this executor's row standardization.
    fn context(&self) -> RowContext<'_> {
        RowContext {
            standardization: self.standardization.as_ref(),
            family: self.family_override.as_deref().unwrap_or(self.ctx.family),
            ..self.ctx
        }
    }

    /// The steps the drivers recorded so far, leaving the trace empty.
    pub fn take_trace(&mut self) -> Vec<SolverStep> {
        std::mem::take(&mut self.trace)
    }

    /// η and μ of every row under `predictor` (in-memory results keep them).
    pub fn materialize(&self, predictor: &Predictor) -> (Vec<f64>, Vec<f64>) {
        let (mut eta, mut mu) = (Vec::new(), Vec::new());
        self.materialize_into(predictor, &mut eta, &mut mu);
        (eta, mu)
    }

    fn materialize_into(&self, predictor: &Predictor, eta: &mut Vec<f64>, mu: &mut Vec<f64>) {
        let n = self.x.nrows();
        let x_slice = self.x.as_slice().expect("contiguous design");
        let p = self.x.ncols();
        eta.resize(n, 0.0);
        mu.resize(n, 0.0);
        // Per-row state: dense rows give the sparse kernels' bits.
        row_states(predictor, &self.context(), 0..n, eta, mu, |i| {
            RowRef::Dense(&x_slice[i * p..(i + 1) * p])
        });
    }

    /// Solve weights (prior weights included) of every row under `weights`.
    pub fn solve_weights(&self, weights: WeightsSpec, predictors: &[Predictor]) -> Vec<f64> {
        let n = self.x.nrows();
        let x_slice = self.x.as_slice().expect("contiguous design");
        let p = self.x.ncols();
        let ctx = self.context();
        let mut scratch = ctx.standardization.map(StandardizedRow::new);
        let mut states = vec![RowState::default(); predictors.len()];
        (0..n)
            .map(|i| {
                let raw = RowRef::Dense(&x_slice[i * p..(i + 1) * p]);
                let row = match (ctx.standardization, scratch.as_mut()) {
                    (Some(standardization), Some(scratch)) => {
                        standardize_row(raw, standardization, scratch)
                    }
                    _ => raw,
                };
                for (state, predictor) in states.iter_mut().zip(predictors.iter()) {
                    *state = row_state(predictor, &ctx, &row, i);
                }
                solve_weight(&ctx, weights, &states, i)
            })
            .collect()
    }

    /// Raw ordinary IRLS weights (without prior weights) at `predictor`.
    pub fn raw_weights(&self, predictor: &Predictor) -> Vec<f64> {
        let n = self.x.nrows();
        let x_slice = self.x.as_slice().expect("contiguous design");
        let p = self.x.ncols();
        let ctx = self.context();
        let mut scratch = ctx.standardization.map(StandardizedRow::new);
        (0..n)
            .map(|i| {
                let raw = RowRef::Dense(&x_slice[i * p..(i + 1) * p]);
                let row = match (ctx.standardization, scratch.as_mut()) {
                    (Some(standardization), Some(scratch)) => {
                        standardize_row(raw, standardization, scratch)
                    }
                    _ => raw,
                };
                irls_weight(&ctx, row_state(predictor, &ctx, &row, i), i)
            })
            .collect()
    }
}

impl PassExecutor for InMemoryExecutor<'_> {
    fn n_rows(&self) -> usize {
        self.x.nrows()
    }

    fn n_params(&self) -> usize {
        if self.intercept_only {
            1
        } else {
            self.x.ncols()
        }
    }

    fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>> {
        let n = self.x.nrows();
        let p = self.x.ncols();
        let x_slice = self.x.as_slice().expect("contiguous design");
        let packed = self.sparse_offsets.as_deref();
        let intercept_only = self.intercept_only;
        let width = if intercept_only { 1 } else { p };
        let ctx = &self.context();
        let total = reduce_rows(PassPartial::zeros(request, width), n, |partial, rows| {
            process_block(
                request,
                ctx,
                x_slice,
                p,
                rows,
                0,
                partial,
                packed,
                intercept_only,
            );
        });
        Ok(total.0)
    }

    fn set_row_standardization(&mut self, standardization: Option<Standardization>) {
        self.standardization = standardization;
    }

    fn set_intercept_only(&mut self, intercept_only: bool) {
        self.intercept_only = intercept_only;
    }

    fn trace(&mut self) -> Option<&mut Vec<SolverStep>> {
        Some(&mut self.trace)
    }

    fn set_family(&mut self, family: Box<dyn Family>) {
        self.family_override = Some(family);
    }

    fn spill_mu(&mut self, predictor: &Predictor) -> Result<()> {
        self.spilled_mu = Some(self.materialize(predictor).1);
        Ok(())
    }

    fn spilled_nb_loglik(&mut self, theta: f64, weighted: bool) -> Result<f64> {
        let mu = self.spilled_mu.as_deref().ok_or_else(|| {
            RustyStatsError::InvalidValue("no μ has been spilled for a θ pass".to_string())
        })?;
        Ok(crate::diagnostics::nb_loglikelihood_slices(
            self.ctx.y,
            mu,
            theta,
            weighted.then_some(self.ctx.prior_weights),
        ))
    }
}

/// The single all-ones column an intercept-only pass reads.
pub(crate) static ONE_COLUMN: [f64; 1] = [1.0];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{BinomialFamily, GammaFamily, PoissonFamily, TweedieFamily};
    use crate::links::{LogLink, LogitLink};
    use crate::solvers::compute_irls_weights;
    use ndarray::{Array1, Array2};

    fn fixture(
        n: usize,
        p: usize,
        sparse: bool,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                1.0
            } else if sparse && (i + j) % 4 != 0 {
                0.0
            } else {
                ((i * 31 + j * 17) % 23) as f64 / 23.0 - 0.4
            }
        });
        let y = Array1::from_shape_fn(n, |i| ((i * 7) % 5) as f64 * 0.5 + 0.25);
        let offset = Array1::from_shape_fn(n, |i| ((i % 3) as f64 - 1.0) * 0.1);
        let prior = Array1::from_shape_fn(n, |i| 0.5 + (i % 4) as f64 * 0.25);
        (x, y, offset, prior)
    }

    fn check(family: &dyn Family, link: &dyn Link, sparse: bool, y_scale: f64) {
        let n = 2 * crate::rows::GRAM_BLOCK_ROWS + 333;
        let p = 17;
        let (x, y0, offset, prior) = fixture(n, p, sparse);
        let y = if y_scale == 0.0 {
            y0.mapv(|v| if v > 0.7 { 1.0 } else { 0.0 })
        } else {
            y0.mapv(|v| v * y_scale)
        };
        let beta: Vec<f64> = (0..p).map(|j| 0.05 * (j as f64 - 8.0) / 8.0).collect();
        let ctx = RowContext {
            family,
            link,
            y: y.as_slice().expect("contiguous test array"),
            offset: offset.as_slice().expect("contiguous test array"),
            prior_weights: prior.as_slice().expect("contiguous test array"),
            y_mean: crate::families::response_mean(&y),
            min_weight: crate::constants::MIN_IRLS_WEIGHT,
            has_offset: true,
            standardization: None,
        };
        let mut exec = InMemoryExecutor::new(x.view(), sparse, ctx).expect("executor");
        let mut request = PassRequest::default();
        let pred = request.predictor(Predictor::coef(&beta));
        request.eval(Eval::Deviance(pred));
        request.eval(Eval::Gram {
            weights: WeightsSpec::Irls(pred),
            response: ResponseSpec::Working(pred),
        });

        let run = |threads: usize, exec: &mut InMemoryExecutor<'_>| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| exec.run(&request).expect("pass"))
        };
        let sums = run(1, &mut exec);
        assert_eq!(sums, run(8, &mut exec), "thread independence");

        // Reference: today's array code at the same state.
        let eta: Array1<f64> = (0..n)
            .map(|i| {
                crate::solvers::irls::row_dot(
                    x.row(i).as_slice().expect("contiguous test array"),
                    &beta,
                ) + offset[i]
            })
            .collect();
        let mu = family.clamp_mu(&link.inverse(&eta));
        let (dev, finite) = sums[0].deviance();
        assert!(finite);
        assert_eq!(
            dev.to_bits(),
            family.deviance(&y, &mu, Some(&prior)).to_bits()
        );

        let weights = compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            family,
            link,
            crate::constants::MIN_IRLS_WEIGHT,
        )
        .expect("weights");
        let (xtx, xtz) = crate::solvers::irls::compute_xtwx_xtwz(
            x.view(),
            &weights.working_response,
            &weights.combined_weights,
        )
        .expect("gram");
        let gram = sums[1].clone().into_gram();
        let mut packed = 0;
        for i in 0..p {
            assert_eq!(gram.xtz[i].to_bits(), xtz[i].to_bits(), "xtz[{i}]");
            for j in i..p {
                assert_eq!(
                    gram.xtx_packed[packed].to_bits(),
                    xtx[(i, j)].to_bits(),
                    "xtx[{i},{j}]"
                );
                packed += 1;
            }
        }
    }

    /// Robust meats from a pass equal `inference::robust_covariance` on the same
    /// state, from dense and sparse rows, at any thread count.
    fn check_robust(family: &dyn Family, link: &dyn Link, sparse: bool) {
        let n = 2 * crate::rows::GRAM_BLOCK_ROWS + 333;
        let p = 17;
        let (x, y0, offset, prior) = fixture(n, p, sparse);
        let y = y0.mapv(|v| v * 2.0);
        let beta: Vec<f64> = (0..p).map(|j| 0.05 * (j as f64 - 8.0) / 8.0).collect();
        // Retained weights at a different predictor than the fitted one, as after
        // a final extraction.
        let beta_weights: Vec<f64> = beta.iter().map(|b| b * 0.9).collect();
        let bread = Array2::from_shape_fn((p, p), |(j, k)| {
            if j == k {
                0.002
            } else {
                0.0001 * ((j + k) % 5) as f64 - 0.0002
            }
        });
        let ctx = RowContext {
            family,
            link,
            y: y.as_slice().expect("contiguous test array"),
            offset: offset.as_slice().expect("contiguous test array"),
            prior_weights: prior.as_slice().expect("contiguous test array"),
            y_mean: crate::families::response_mean(&y),
            min_weight: crate::constants::MIN_IRLS_WEIGHT,
            has_offset: true,
            standardization: None,
        };
        let types = vec![HCType::HC0, HCType::HC1, HCType::HC2, HCType::HC3];
        let mut exec = InMemoryExecutor::new(x.view(), sparse, ctx).expect("executor");
        let mut request = PassRequest::default();
        let fitted = request.predictor(Predictor::coef(&beta));
        let weights = request.predictor(Predictor::coef(&beta_weights));
        request.eval(Eval::RobustMeats {
            predictor: fitted,
            weights: WeightsSpec::Irls(weights),
            types: types.clone(),
            bread: bread.iter().copied().collect(),
            hc1: crate::inference::hc1_scale(n, p),
            raw_rows: false,
        });
        let run = |threads: usize, exec: &mut InMemoryExecutor<'_>| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| exec.run(&request).expect("pass"))
        };
        let sums = run(1, &mut exec);
        assert_eq!(sums, run(8, &mut exec), "thread independence");

        let state = |coef: &[f64]| {
            let eta: Array1<f64> = (0..n)
                .map(|i| {
                    crate::solvers::irls::row_dot(
                        x.row(i).as_slice().expect("contiguous test array"),
                        coef,
                    ) + offset[i]
                })
                .collect();
            let mu = family.clamp_mu(&link.inverse(&eta));
            (eta, mu)
        };
        let (_, mu) = state(&beta);
        let (eta_w, mu_w) = state(&beta_weights);
        let raw = compute_irls_weights(
            &y,
            &mu_w,
            &eta_w,
            &offset,
            &prior,
            family,
            link,
            crate::constants::MIN_IRLS_WEIGHT,
        )
        .expect("weights")
        .irls_weights;
        let resid = crate::diagnostics::resid_pearson(&y, &mu, family);
        for (hc_type, meat) in types.iter().zip(sums[0].meats()) {
            let from_pass = crate::inference::sandwich_from_packed_meat(&bread, &meat.xtx_packed);
            let expected =
                crate::inference::robust_covariance(&x, &resid, &raw, &prior, &bread, *hc_type);
            let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&from_pass),
                bits(&expected),
                "{hc_type:?}, sparse={sparse}"
            );
        }
    }

    /// The reconstruct pass names itself; the same evaluations otherwise name a Gram
    /// pass, and fold filters win (§4.4).
    #[test]
    fn the_reconstruct_pass_reports_its_kind() {
        let beta = [0.1, 0.2];
        let gram = |request: &mut PassRequest| {
            let index = request.predictor(Predictor::coef(&beta));
            request.eval(Eval::Gram {
                weights: WeightsSpec::Irls(index),
                response: ResponseSpec::Working(index),
            });
        };
        let mut plain = PassRequest::default();
        gram(&mut plain);
        assert_eq!(plain.kind(), "gram");
        let mut reconstruct = PassRequest::reconstruct();
        gram(&mut reconstruct);
        assert_eq!(reconstruct.kind(), "reconstruct");
        reconstruct.filters[0] = Some(FoldFilter {
            fold: 0,
            folds: 3,
            seed: 1,
            validation: false,
        });
        assert_eq!(reconstruct.kind(), "fold");
    }

    /// The kernel choice reads n, p and the structural bound only (§5.2), with
    /// today's thresholds; a block's CSR rows are its dense rows' nonzeros.
    #[test]
    fn kernel_dispatch_is_structural_and_block_rows_are_exact() {
        assert!(!sparse_rows_for(0, 100, 0));
        assert!(!sparse_rows_for(1_000_000, 15, 0));
        assert!(!sparse_rows_for(99_999, 100, 1), "below the minimum cells");
        assert!(sparse_rows_for(100_000, 100, 35));
        assert!(!sparse_rows_for(100_000, 100, 36));
        // No cap on the design's total nonzeros: the CSR is per block.
        assert!(sparse_rows_for(50_000_000, 100, 35));

        // Out of core: no structural bound means dense rows; an explicit kernel wins.
        let offsets_100 = Some(crate::solvers::irls::packed_upper_offsets(100));
        assert_eq!(sparse_row_offsets(100_000, 100, RowKernel::default()), None);
        let bounded = RowKernel::Structural(Some(10));
        assert_eq!(sparse_row_offsets(100_000, 100, bounded), offsets_100);
        assert_eq!(sparse_row_offsets(100_000, 100, RowKernel::Dense), None);
        assert_eq!(sparse_row_offsets(10, 100, RowKernel::Sparse), offsets_100);

        // In memory: the structural bound by the shape alone (a broadcast view holds
        // no rows), X's own rows without one, and an explicit kernel whatever X is.
        let zero = Array1::<f64>::zeros(1);
        let wide = zero.broadcast((100_000, 100)).expect("view");
        assert!(RowKernel::Structural(Some(35)).sparse_in_memory(wide));
        assert!(!RowKernel::Structural(Some(36)).sparse_in_memory(wide));
        assert!(
            !RowKernel::Structural(None).sparse_in_memory(wide),
            "no rows to scan"
        );
        let small = Array2::<f64>::zeros((10, 20));
        assert!(!RowKernel::Structural(None).sparse_in_memory(small.view()));
        assert!(!RowKernel::Structural(Some(0)).sparse_in_memory(small.view()));
        assert!(RowKernel::Sparse.sparse_in_memory(small.view()));
        assert!(!RowKernel::Dense.sparse_in_memory(wide));
        // The scan reads every row: one dense row makes the design dense.
        let mut scanned = Array2::<f64>::zeros((250_000, 40));
        scanned.column_mut(0).fill(1.0);
        assert!(RowKernel::Structural(None).sparse_in_memory(scanned.view()));
        scanned.row_mut(249_999).fill(2.0);
        assert!(!RowKernel::Structural(None).sparse_in_memory(scanned.view()));
        assert_eq!(design_max_row_nonzeros(&[], 0), 0);
        let x = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, -0.0, 4.0, 5.0, 0.0, 6.0, 0.0];
        assert_eq!(design_max_row_nonzeros(&x, 3), 2);
        let offsets = crate::solvers::irls::packed_upper_offsets(3);
        let csr = BlockCsr::build(&x, 3, 1..4);
        // Allocated at the block's exact nonzero count (§11's CSR term).
        assert_eq!((csr.indices.len(), csr.values.len()), (3, 3));
        assert_eq!(csr.indices.capacity(), 3);
        assert_eq!(csr.values.capacity(), 3);
        assert_eq!(csr.offsets.capacity(), 4);
        for (row, (indices, values)) in [
            (1, (vec![], vec![])),
            (2, (vec![1u32, 2], vec![4.0, 5.0])),
            (3, (vec![1u32], vec![6.0])),
        ] {
            match csr.row(row, &offsets) {
                RowRef::Sparse {
                    indices: got_indices,
                    values: got_values,
                    ..
                } => {
                    assert_eq!(got_indices, indices.as_slice(), "row {row}");
                    assert_eq!(got_values, values.as_slice(), "row {row}");
                }
                RowRef::Dense(_) => panic!("a CSR row is sparse"),
            }
        }
    }

    /// A row standardized in the executor gives the bits of the same row of
    /// `standardize_matrix`'s copy, from dense and from sparse rows, for a
    /// scale-only and a centred standardization.
    #[test]
    fn standardized_rows_match_a_standardized_copy_bitwise() {
        let n = crate::rows::GRAM_BLOCK_ROWS + 777;
        let p = 17;
        let (x, y0, offset, prior) = fixture(n, p, true);
        let y = y0.mapv(|v| v * 2.0);
        let beta: Vec<f64> = (0..p).map(|j| 0.03 * (j as f64 - 8.0) / 8.0).collect();
        let scale: Vec<f64> = (0..p)
            .map(|j| {
                if j % 3 == 0 {
                    1.0
                } else {
                    0.5 + j as f64 * 0.1
                }
            })
            .collect();
        let scale_only = Standardization::new(vec![0.0; p], scale.clone()).expect("valid");
        let centred = Standardization::new(
            (0..p)
                .map(|j| if j % 2 == 1 { 0.05 * j as f64 } else { 0.0 })
                .collect(),
            scale,
        )
        .expect("valid");
        let bread = Array2::from_shape_fn((p, p), |(j, k)| if j == k { 0.002 } else { 0.0001 });

        let mut request = PassRequest::default();
        let index = request.predictor(Predictor::coef(&beta));
        request.eval(Eval::Deviance(index));
        request.eval(Eval::Gram {
            weights: WeightsSpec::Irls(index),
            response: ResponseSpec::Working(index),
        });
        request.eval(Eval::RobustMeats {
            predictor: index,
            weights: WeightsSpec::Irls(index),
            types: vec![HCType::HC3],
            bread: bread.iter().copied().collect(),
            hc1: crate::inference::hc1_scale(n, p),
            raw_rows: false,
        });
        let ctx = || RowContext {
            family: &PoissonFamily,
            link: &LogLink,
            y: y.as_slice().expect("contiguous test array"),
            offset: offset.as_slice().expect("contiguous test array"),
            prior_weights: prior.as_slice().expect("contiguous test array"),
            y_mean: crate::families::response_mean(&y),
            min_weight: crate::constants::MIN_IRLS_WEIGHT,
            has_offset: true,
            standardization: None,
        };
        for standardization in [&scale_only, &centred] {
            let copy = standardization.standardize_matrix(x.view()).expect("copy");
            let mut reference = InMemoryExecutor::new(copy.view(), false, ctx()).expect("executor");
            let expected = reference.run(&request).expect("pass");
            for sparse in [false, true] {
                let mut executor =
                    InMemoryExecutor::new(x.view(), sparse, ctx()).expect("executor");
                executor.set_row_standardization(Some(standardization.clone()));
                assert_eq!(
                    executor.run(&request).expect("pass"),
                    expected,
                    "sparse={sparse}"
                );
                let materialized = executor.materialize(&Predictor::coef(&beta));
                let reference_state = reference.materialize(&Predictor::coef(&beta));
                assert_eq!(materialized, reference_state);
            }
        }
    }

    /// The monotone Gram equals the Gram of an explicitly transformed design
    /// (scale × reverse cumulative sums per term) with the adjusted working
    /// response, from dense and sparse rows; a partitioned predictor's η is the
    /// refit columns' dot plus the fixed columns' offset.
    #[test]
    fn monotone_gram_and_partitioned_predictor_match_explicit_arithmetic() {
        let n = crate::rows::GRAM_BLOCK_ROWS + 211;
        let p = 17;
        let (x, y0, offset, prior) = fixture(n, p, true);
        let y = y0.mapv(|v| v * 2.0);
        let beta: Vec<f64> = (0..p).map(|j| 0.03 * (j as f64 - 8.0) / 8.0).collect();
        let transform = MonotoneTransform {
            terms: vec![
                MonotoneTermTransform {
                    start: 3,
                    end: 7,
                    scales: vec![1.0, 0.5, 1.5, 0.25],
                    offset: vec![0.1, -0.2, 0.3, 0.05],
                },
                MonotoneTermTransform {
                    start: 10,
                    end: 13,
                    scales: vec![1.0, -0.75, -2.0],
                    offset: vec![-0.4, 0.2, 0.1],
                },
            ],
        };
        let ctx = || RowContext {
            family: &PoissonFamily,
            link: &LogLink,
            y: y.as_slice().expect("contiguous test array"),
            offset: offset.as_slice().expect("contiguous test array"),
            prior_weights: prior.as_slice().expect("contiguous test array"),
            y_mean: crate::families::response_mean(&y),
            min_weight: crate::constants::MIN_IRLS_WEIGHT,
            has_offset: true,
            standardization: None,
        };
        let mut request = PassRequest::default();
        let index = request.predictor(Predictor::coef(&beta));
        request.eval(Eval::MonotoneGram {
            predictor: index,
            transform: Arc::new(transform.clone()),
        });
        let refit: Arc<[usize]> = [0usize, 1, 2, 8, 9, 14, 15, 16].into();
        let fixed: Arc<[(usize, f64)]> = [(3usize, 0.2), (5, -0.1), (11, 0.05)].into();
        let partitioned = request.predictor(Predictor::Partitioned {
            refit: refit.clone(),
            beta: refit.iter().map(|&c| beta[c]).collect(),
            fixed: fixed.clone(),
        });
        request.eval(Eval::Deviance(partitioned));

        // Reference: the transformed copy of X, the adjusted response at β.
        let mut x_tilde = x.clone();
        let mut z_adjusted = Array1::zeros(n);
        let mut weights = Array1::zeros(n);
        let mut deviance_terms = Vec::with_capacity(n);
        for i in 0..n {
            let row = x.row(i);
            let eta =
                crate::solvers::irls::row_dot(row.as_slice().expect("row"), &beta) + offset[i];
            let mu = PoissonFamily.clamp_mu_at(LogLink.inverse_at(eta));
            let d = LogLink.derivative_at(mu);
            // max/min, not clamp: the kernels' NaN handling.
            #[allow(clippy::manual_clamp)]
            let irls = (1.0 / (PoissonFamily.variance_at(mu) * d * d))
                .max(crate::constants::MIN_IRLS_WEIGHT)
                .min(MAX_IRLS_WEIGHT);
            weights[i] = prior[i] * irls;
            let mut z = (eta - offset[i]) + (y[i] - mu) * d;
            for term in &transform.terms {
                let mut dot = 0.0;
                for (c, &o) in (term.start..term.end).zip(term.offset.iter()) {
                    dot += row[c] * o;
                }
                z -= dot;
            }
            z_adjusted[i] = z;
            for term in &transform.terms {
                let mut cumsum = 0.0;
                for c in (term.start..term.end).rev() {
                    cumsum += row[c];
                    x_tilde[[i, c]] = term.scales[c - term.start] * cumsum;
                }
            }
            let mut dot = 0.0;
            for (&c, b) in refit.iter().zip(refit.iter().map(|&c| beta[c])) {
                dot += row[c] * b;
            }
            let mut o_f = offset[i];
            for &(c, coef) in fixed.iter() {
                o_f += coef * row[c];
            }
            let mu_p = PoissonFamily.clamp_mu_at(LogLink.inverse_at(dot + o_f));
            deviance_terms.push(PoissonFamily.unit_deviance_at(y[i], mu_p) * prior[i]);
        }
        let (xtx, xtz) =
            crate::solvers::irls::compute_xtwx_xtwz(x_tilde.view(), &z_adjusted, &weights)
                .expect("gram");
        let expected_deviance = crate::rows::reduce_rows(0.0, n, |sum: &mut f64, rows| {
            for i in rows {
                *sum += deviance_terms[i];
            }
        });

        for sparse in [false, true] {
            let mut executor = InMemoryExecutor::new(x.view(), sparse, ctx()).expect("executor");
            let sums = executor.run(&request).expect("pass");
            let gram = sums[0].clone().into_gram();
            let mut packed = 0;
            for i in 0..p {
                assert_eq!(gram.xtz[i].to_bits(), xtz[i].to_bits(), "xtz[{i}]");
                for j in i..p {
                    assert_eq!(gram.xtx_packed[packed].to_bits(), xtx[(i, j)].to_bits());
                    packed += 1;
                }
            }
            assert_eq!(
                sums[1].deviance().0.to_bits(),
                expected_deviance.to_bits(),
                "partitioned deviance, sparse={sparse}"
            );
        }
    }

    #[test]
    fn robust_meats_match_robust_covariance_bitwise() {
        check_robust(&PoissonFamily, &LogLink, false);
        check_robust(&PoissonFamily, &LogLink, true);
        check_robust(&GammaFamily, &LogLink, true);
        check_robust(&TweedieFamily::new(1.5).expect("tweedie"), &LogLink, false);
    }

    #[test]
    fn pass_kernel_matches_array_code_bitwise() {
        check(&PoissonFamily, &LogLink, false, 2.0);
        check(&PoissonFamily, &LogLink, true, 2.0);
        check(&GammaFamily, &LogLink, false, 3.0);
        check(
            &TweedieFamily::new(1.5).expect("tweedie"),
            &LogLink,
            true,
            3.0,
        );
        check(&BinomialFamily, &LogitLink, false, 0.0);
    }
}
