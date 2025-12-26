# Mathematical Notation

This page defines the mathematical notation used throughout the documentation.

## Variables and Data

| Symbol | Description |
|--------|-------------|
| \(n\) | Number of observations |
| \(p\) | Number of parameters (including intercept) |
| \(Y\), \(y\) | Response variable (random/observed) |
| \(y_i\) | Response for observation \(i\) |
| \(\mathbf{y}\) | Response vector \((y_1, \ldots, y_n)^T\) |
| \(X\), \(\mathbf{X}\) | Design matrix (\(n \times p\)) |
| \(\mathbf{x}_i\) | Row vector of predictors for observation \(i\) |
| \(x_{ij}\) | Value of predictor \(j\) for observation \(i\) |

## Parameters

| Symbol | Description |
|--------|-------------|
| \(\boldsymbol{\beta}\) | Coefficient vector \((\beta_0, \beta_1, \ldots, \beta_{p-1})^T\) |
| \(\beta_0\) | Intercept |
| \(\beta_j\) | Coefficient for predictor \(j\) |
| \(\hat{\boldsymbol{\beta}}\) | Estimated coefficients |
| \(\phi\) | Dispersion parameter |
| \(\hat{\phi}\) | Estimated dispersion |

## GLM Components

### Linear Predictor

| Symbol | Description |
|--------|-------------|
| \(\eta\) | Linear predictor |
| \(\eta_i\) | Linear predictor for observation \(i\): \(\eta_i = \mathbf{x}_i^T \boldsymbol{\beta}\) |
| \(\boldsymbol{\eta}\) | Vector of linear predictors |

### Mean and Link

| Symbol | Description |
|--------|-------------|
| \(\mu\) | Mean: \(E(Y)\) |
| \(\mu_i\) | Mean for observation \(i\): \(\mu_i = E(Y_i)\) |
| \(\hat{\mu}_i\) | Fitted mean |
| \(g(\cdot)\) | Link function |
| \(g^{-1}(\cdot)\) | Inverse link function (mean function) |
| \(g'(\mu)\) | Derivative of link function |

The relationship:
\[
\eta = g(\mu) \quad \Leftrightarrow \quad \mu = g^{-1}(\eta)
\]

### Variance Function

| Symbol | Description |
|--------|-------------|
| \(V(\mu)\) | Variance function |
| \(\text{Var}(Y)\) | Variance of response: \(\text{Var}(Y) = \phi \cdot V(\mu)\) |

## Matrices and Vectors

| Symbol | Description |
|--------|-------------|
| \(\mathbf{I}\) | Identity matrix |
| \(\mathbf{W}\) | Diagonal weight matrix |
| \(W_{ii}\) | Weight for observation \(i\) |
| \(\mathbf{z}\) | Working response vector |
| \((\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\) | Unscaled covariance matrix |

## Likelihood and Deviance

| Symbol | Description |
|--------|-------------|
| \(L\) | Likelihood |
| \(\ell\) | Log-likelihood: \(\ell = \log L\) |
| \(D\) | Deviance |
| \(d_i\) | Unit deviance for observation \(i\) |
| \(D_{\text{null}}\) | Null deviance (intercept-only model) |

Deviance definition:
\[
D = 2[\ell(\text{saturated}) - \ell(\text{fitted})] = \sum_{i=1}^n d_i
\]

## Regularization

| Symbol | Description |
|--------|-------------|
| \(\alpha\) | Overall penalty strength |
| \(\rho\) | L1 ratio (1 = Lasso, 0 = Ridge) |
| \(\lambda\) | Alternative notation for penalty strength |
| \(\|\boldsymbol{\beta}\|_1\) | L1 norm: \(\sum_j |\beta_j|\) |
| \(\|\boldsymbol{\beta}\|_2^2\) | Squared L2 norm: \(\sum_j \beta_j^2\) |

Elastic Net penalty:
\[
P(\boldsymbol{\beta}) = \alpha \left[ \rho \|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\beta}\|_2^2 \right]
\]

## Inference

| Symbol | Description |
|--------|-------------|
| \(\text{SE}(\hat{\beta}_j)\) | Standard error of \(\hat{\beta}_j\) |
| \(z\) | z-statistic: \(z = \hat{\beta} / \text{SE}(\hat{\beta})\) |
| \(\alpha\) | Significance level (context-dependent) |
| \(\text{CI}\) | Confidence interval |

Standard error:
\[
\text{SE}(\hat{\beta}_j) = \sqrt{\hat{\phi} \cdot [(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}]_{jj}}
\]

## Residuals

| Symbol | Description |
|--------|-------------|
| \(r_i^{\text{response}}\) | Response residual: \(y_i - \hat{\mu}_i\) |
| \(r_i^{\text{Pearson}}\) | Pearson residual: \(\frac{y_i - \hat{\mu}_i}{\sqrt{V(\hat{\mu}_i)}}\) |
| \(r_i^{\text{deviance}}\) | Deviance residual: \(\text{sign}(y_i - \hat{\mu}_i)\sqrt{d_i}\) |
| \(r_i^{\text{working}}\) | Working residual: \((y_i - \hat{\mu}_i) \cdot g'(\hat{\mu}_i)\) |

## Information Criteria

| Symbol | Formula |
|--------|---------|
| AIC | \(-2\ell + 2p\) |
| BIC | \(-2\ell + p \log(n)\) |

## Common Link Functions

| Name | \(g(\mu)\) | \(g^{-1}(\eta)\) | \(g'(\mu)\) |
|------|-----------|------------------|-------------|
| Identity | \(\mu\) | \(\eta\) | \(1\) |
| Log | \(\log(\mu)\) | \(e^\eta\) | \(1/\mu\) |
| Logit | \(\log\frac{\mu}{1-\mu}\) | \(\frac{e^\eta}{1+e^\eta}\) | \(\frac{1}{\mu(1-\mu)}\) |
| Probit | \(\Phi^{-1}(\mu)\) | \(\Phi(\eta)\) | \(\frac{1}{\phi(\Phi^{-1}(\mu))}\) |

## Common Variance Functions

| Family | \(V(\mu)\) |
|--------|-----------|
| Gaussian | \(1\) |
| Poisson | \(\mu\) |
| Binomial | \(\mu(1-\mu)\) |
| Gamma | \(\mu^2\) |
| Tweedie | \(\mu^p\) |
| Negative Binomial | \(\mu + \mu^2/\theta\) |

## IRLS Algorithm

Working weights:
\[
W_{ii} = \frac{1}{V(\mu_i) \cdot [g'(\mu_i)]^2}
\]

Working response:
\[
z_i = \eta_i + (y_i - \mu_i) \cdot g'(\mu_i)
\]

Update step:
\[
\boldsymbol{\beta}^{(t+1)} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{z}
\]

## Splines

| Symbol | Description |
|--------|-------------|
| \(B_{i,k}(x)\) | B-spline basis function of degree \(k\) |
| \(t_0, t_1, \ldots\) | Knot sequence |
| df | Degrees of freedom (number of basis functions) |

Cox-de Boor recursion:
\[
B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)
\]

## Probability Distributions

| Distribution | Notation | Parameters |
|--------------|----------|------------|
| Normal | \(N(\mu, \sigma^2)\) | mean \(\mu\), variance \(\sigma^2\) |
| Poisson | \(\text{Pois}(\lambda)\) | rate \(\lambda\) |
| Binomial | \(\text{Bin}(n, p)\) | trials \(n\), probability \(p\) |
| Gamma | \(\text{Gamma}(\alpha, \beta)\) | shape \(\alpha\), rate \(\beta\) |
| Negative Binomial | \(\text{NB}(\mu, \theta)\) | mean \(\mu\), dispersion \(\theta\) |
