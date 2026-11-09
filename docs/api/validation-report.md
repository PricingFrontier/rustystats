# Validation Report

`GLMModel.html_report()` validates a fitted GLM and returns one self-contained HTML page: what a
reviewer needs to decide whether the model is ready for use, in and out of sample.

```python
html = result.html_report(train, test_data=test, name="Motor frequency v3")
Path("report.html").write_text(html)
```

The page has no scripts and loads nothing from the network, so it opens offline and displays in a
sandboxed viewer such as MLflow's artifact viewer. It has the same layout and theme as the
[t-boost](https://github.com/PricingFrontier/t-boost) model report, so the two can be read side
by side.

The report and the page are separate steps, so the record can be kept beside the model:

```python
import json
import rustystats as rs

report = result.validation_report(train, test_data=test, name="Motor frequency v3")
with open("freq_report.json", "w") as f:
    json.dump(report, f)
html = rs.report_html(report)          # draws the report; computes nothing
```

For MLflow, `report["summary_metrics"]` is a flat dictionary of finite values under valid metric
names:

```python
mlflow.log_metrics(report["summary_metrics"])
mlflow.log_dict(report, "rustystats/report.json")
mlflow.log_text(rs.report_html(report), "rustystats/report.html")
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `train_data` | required | The data the model was fitted on |
| `test_data` | `None` | Held-out data for the out-of-sample columns and charts |
| `name` | `None` | The heading of the page |
| `thresholds` | `None` | Overrides of the review thresholds; `None` as a value turns a check off |
| `categorical_factors`, `continuous_factors` | `None` | Further columns to review; those not in the model get a score test |
| `interactions` | `None` | Pairs of factors to draw as A/E grids, as `(a, b)`, beside the model's fitted interactions |
| `base_predictions` | `None` | Predictions from another model, compared against this one |
| `n_factor_bins` | `10` | Quantile bands for each continuous factor |
| `exposure`, `weights` | `None` | Overrides of the fitted exposure and weights |
| `labels` | `("train", "test")` | The names of the two datasets on the page and in the metric keys |

The model's own factors are read from its terms, so the one-line call needs no factor lists.
Like `diagnostics()`, the report writes `analysis/diagnostics.json` to the working directory.

## Multinomial models

`MultinomialModel` has the same three calls, and the page adds a class switch beside the dataset
switch:

```python
model = rs.multinomial_dict(response="tier", terms=terms, data=train, reference="basic").fit()
html = model.html_report(train, test_data=test, name="Tier choice")
```

The report takes the same arguments as the GLM report, except `exposure` and `base_predictions`.
Each class is reviewed one-vs-rest: its predicted probability against whether the row was that
class. Every chart, band and review flag is then per class, and a flag names its class.

Beside the per-class views, the page shows what belongs to the model as a whole: deviance against
the class shares (McFadden's deviance explained), log loss, accuracy, macro F1, the expected-count
and most-likely-class matrices, and the actual and predicted class mix. Fitted effects are odds
against the reference class, which has no coefficients of its own. Factor importance comes from
each factor's joint Wald test across the classes.

Coefficients shared by every class (alternative-specific attributes) are listed on the
Coefficients tab. Attributes that vary by alternative are not factors of a row, so they have no
Factors page, and leverage is not computed.

## The page

A bar of tabs, and one dataset switch that selects the dataset drawn in colour in every chart;
the others are drawn in grey. Blue is observed data and orange is the model. Each chart has a
one-line caption and its numbers collapsed beneath it; hovering a point shows its values.

**Overview.** One screen: deviance explained, Gini and A/E of each dataset; the open review
flags; calibration by decile; each factor's share of the explained deviance; the largest fitted
effects; and the variance check.

**Performance.** Metrics per dataset (deviance, null deviance, deviance explained, Gini, AIC,
BIC); calibration by decile with ±2σ; the Lorenz curve; every factor's A/E by band at a glance;
the spread of the predictions; and, with `base_predictions`, a comparison with the base model.

**Factors.** One figure per factor on one x-axis: the fitted effect with its 95% band, the
actual and expected rate, the A/E of each dataset by band with ±2σ, and the exposure. Partial
dependence is added when the factor is also in an interaction. Factors passed in but not in the
model are listed with their score test. Numbers by band, relativities and coefficients are
collapsed below.

**Interactions.** A/E by cell of each pair of factors, with the fitted interaction surface for
pairs in the model and any detected candidates. Structure in the A/E of a pair not in the model
is the evidence for adding it.

**Coefficients.** A forest plot of the estimates with their 95% intervals, and the full table.

**Fit.** Goodness of fit and convergence; the variance check and A/E by band of fitted value;
residuals; collinearity (GVIF and the largest correlations between estimates); influence
(leverage and Cook's distance); the model's terms; smooth terms and the regularisation path
where they apply.

**Checks.** Every review flag with its status and threshold, the data checks (missing and unseen
values, population stability), the model's details and the definition of each statistic.

## Reading the statistics

**A/E and z.** A band's z is `(actual − expected) / σ`, where `σ²` sums `φ · weight · V(μ)` over
the band's rows. A band is flagged when it is off by more than `factor_ae` (10%) *and* `factor_z`
(3) standard deviations, so a small band's noise does not raise a warning. Deciles, interaction
cells and the overall A/E follow the same rule.

**The train A/E of a fitted categorical factor is 1 by construction.** The score equations of a
canonical-link fit balance each level. Read the test data for those factors.

**Dispersion.** The page uses `Σ w (y − μ)² / Σ w V(μ)`, the variance of a total against what the
family assumes. The row-level Pearson statistic is also shown, but it weights rows by `1 / V(μ)`,
so on low-frequency data a few rows with a very small expected value inflate it.

**The variance check.** The same ratio by band of fitted value. Flat means the family's `V(μ)`
fits; rising means the variance grows faster than the family assumes.

**No normal QQ plot for counts.** At a mean of 0.05 a QQ plot of residuals looks broken however
good the fit. Count families show observed and expected counts instead; a QQ plot is drawn for
the Gaussian and Gamma families.

**Fitted effects** cover the central 99% of the exposure. Linear and spline terms run away in the
sparse far tail, and would otherwise set the scale of the whole chart. Confidence bands are left
out where the standard errors are not valid, as after regularisation or with smooth terms.

## Thresholds

| Threshold | Default | Check |
|---|---|---|
| `ae_train` | 0.02 | Overall A/E on the train data within 1 ± value |
| `ae_test` | 0.05 | Overall A/E on the test data within 1 ± value |
| `decile_ae` | 0.05 | Fitted-value deciles off by more than value (and `factor_z`) |
| `factor_ae` | 0.10 | Factor bands off by more than value |
| `factor_min_mass` | 0.01 | Only bands holding at least this share of the mass |
| `factor_z` | 3.0 | Only bands at least this many standard deviations out (`None`: the tolerance alone) |
| `near_miss_z` | 2.0 | Past the tolerance with z of at least value: listed as a near miss |
| `interaction_ae` | 0.15 | Interaction cells off by more than value (and `factor_z`) |
| `train_gap` | 0.05 | Drop in deviance explained from train to test |
| `psi_warn`, `psi_critical` | 0.10, 0.25 | Population stability per factor |
| `unseen_mass` | 0.01 | Mass share of levels the train data did not have |
| `dispersion` | 1.2 | Dispersion above value on a family that fixes it at 1 |
| `gvif` | 10 | Variance inflation per coefficient, `GVIF^(1/df)` |
| `term_p` | 0.05 | Fitted factors whose joint test has a p-value above value |

The flags support a review; they are not a verdict. The values in force are printed beside every
flag and listed under Checks.

## Limits

The report cannot verify which rows were used to fit the model and which were held out. It covers
GLMs from `glm_dict` and the formula API and models from `multinomial_dict`, fitted in memory, out
of core or loaded from a file. A GLM loaded with `from_bytes` holds no covariance, so the page
leaves out its standard errors, coefficient table and influence. An out-of-core fit keeps no rows,
so the page leaves out influence and robust standard errors; pass the data it was fitted on.
