# Glossary

Terms and concepts used throughout RustyStats documentation.

---

## A

### AIC (Akaike Information Criterion)
A measure of model quality balancing fit and complexity: AIC = -2×loglik + 2p. Lower is better.

### Alpha (α)
Regularization strength parameter. Higher α means more penalty.

---

## B

### BIC (Bayesian Information Criterion)
Similar to AIC but with stronger penalty for complexity: BIC = -2×loglik + p×log(n).

### B-spline
Piecewise polynomial basis functions defined by knots. See [Splines](../components/splines.md).

### Binomial Family
Distribution for binary (0/1) or proportion data. Variance: V(μ) = μ(1-μ).

---

## C

### Canonical Link
The "natural" link function for a family, derived from the exponential family form.

### Coordinate Descent
Optimization algorithm that updates one variable at a time. Used for regularized GLMs.

### Cross-Validation (CV)
Technique for model selection by training on subsets and testing on holdout data.

---

## D

### Design Matrix
The matrix X containing predictor values. Each row is an observation, each column a feature.

### Deviance
A measure of model fit: D = 2×[ℓ(saturated) - ℓ(fitted)]. Lower is better.

### Dispersion (φ)
Scale parameter relating variance to the variance function: Var(Y) = φ×V(μ).

### Dummy Coding
Representing a categorical variable with k levels using k-1 binary columns.

---

## E

### Elastic Net
Regularization combining L1 and L2 penalties: α×[ρ×‖β‖₁ + (1-ρ)/2×‖β‖₂²].

### Exponential Family
Class of probability distributions with specific form, including Gaussian, Poisson, Binomial.

---

## F

### Family
The distribution of the response variable in a GLM. See [Families](../theory/families.md).

### Fisher Scoring
Variant of Newton-Raphson using expected Hessian. Equivalent to IRLS for GLMs.

### Fitted Values
Predicted means μ̂ = g⁻¹(Xβ̂).

---

## G

### Gamma Family
Distribution for positive continuous data. Variance: V(μ) = μ².

### Gaussian Family
Normal distribution. Variance: V(μ) = 1 (constant).

### Gini Coefficient
Measure of model discrimination: Gini = 2×AUC - 1. Range [0, 1].

### GLM (Generalized Linear Model)
Extension of linear regression allowing different response distributions and link functions.

---

## H

### HC0, HC1, HC2, HC3
Heteroscedasticity-consistent standard error estimators. See [Robust SE](../theory/diagnostics.md).

### Hosmer-Lemeshow Test
Statistical test for model calibration.

---

## I

### Identity Link
Link function g(μ) = μ. Used with Gaussian family.

### Intercept
The constant term β₀ in the linear predictor.

### IRLS (Iteratively Reweighted Least Squares)
Algorithm for fitting GLMs. See [IRLS](../theory/irls.md).

---

## L

### L1 Ratio
Parameter controlling Elastic Net mix: 1.0 = Lasso, 0.0 = Ridge.

### Lasso
L1-penalized regression that performs variable selection: min D(β) + α‖β‖₁.

### Linear Predictor (η)
The linear combination η = Xβ + offset.

### Link Function
Function g(μ) connecting mean to linear predictor. See [Links](../theory/links.md).

### Log Link
Link function g(μ) = log(μ). Ensures positive predictions.

### Logit Link
Link function g(μ) = log(μ/(1-μ)). Maps (0,1) to ℝ.

### Log-Likelihood
The log of the probability of observing the data given the model.

---

## M

### Maximum Likelihood Estimation (MLE)
Finding parameters that maximize the likelihood of the data.

---

## N

### Natural Spline
B-spline with additional constraints for linear extrapolation at boundaries.

### Negative Binomial
Distribution for overdispersed counts. Variance: V(μ) = μ + μ²/θ.

### Newton-Raphson
Iterative optimization algorithm using gradient and Hessian.

### Null Deviance
Deviance of the intercept-only model.

---

## O

### Odds Ratio
exp(β) in logistic regression. Multiplicative effect on odds.

### Offset
Known component added to linear predictor but not estimated. Example: log(exposure).

### Overdispersion
When observed variance exceeds model-predicted variance.

---

## P

### P-value
Probability of observing test statistic as extreme as computed, under null hypothesis.

### Pearson Residual
Standardized residual: (y - μ) / √V(μ).

### Poisson Family
Distribution for count data. Variance: V(μ) = μ.

### Prior Weight
Known weight for each observation, used in likelihood.

### Probit
Link function using inverse normal CDF: g(μ) = Φ⁻¹(μ).

### PyO3
Rust library for Python bindings.

---

## Q

### QuasiPoisson / QuasiBinomial
Quasi-likelihood families that estimate dispersion from data.

---

## R

### Rayon
Rust library for data parallelism.

### Reference Level
The omitted category in dummy coding (absorbed into intercept).

### Regularization
Adding penalty to prevent overfitting. See [Regularization](../theory/regularization.md).

### Relativity
exp(β) for log-link models. Multiplicative effect on mean.

### Residual
Difference between observed and predicted: various types exist.

### Ridge
L2-penalized regression: min D(β) + α‖β‖₂².

### Robust Standard Errors
Standard errors that account for model misspecification.

---

## S

### Sandwich Estimator
Robust covariance estimator: (X'WX)⁻¹ × Meat × (X'WX)⁻¹.

### Score Function
Derivative of log-likelihood with respect to parameters.

### Soft Thresholding
Operation used in Lasso: S(z,γ) = sign(z)×max(|z|-γ, 0).

### Spline
Piecewise polynomial function. See [Splines](../components/splines.md).

### Standard Error (SE)
Estimated standard deviation of a parameter estimate.

---

## T

### Target Encoding
Replacing categories with target-based statistics. See [Target Encoding](../components/target-encoding.md).

### Theta (θ)
Negative Binomial dispersion parameter. Larger θ = less overdispersion.

### Tweedie
Family with variance V(μ) = μᵖ. Useful for insurance pure premiums.

---

## U

### Unit Deviance
Per-observation contribution to total deviance.

---

## V

### Variance Function
V(μ) defining how variance relates to mean: Var(Y) = φ×V(μ).

### Variance Power (p)
Parameter in Tweedie family: V(μ) = μᵖ.

---

## W

### Wald Test
Hypothesis test using (β̂/SE)² as test statistic.

### Weights
Prior observation weights or IRLS working weights.

### Working Response
Linearized response in IRLS: z = η + (y - μ)×g'(μ).

### Working Weights
IRLS weights: W = 1/[V(μ)×g'(μ)²].

---

## Z

### z-statistic
Test statistic: z = β̂ / SE(β̂). Used for hypothesis testing.
