# RustyStats Documentation

**High-performance Generalized Linear Models with a Rust backend and Python API**

RustyStats is a statistical modeling library designed for actuarial and data science applications. It combines the performance of Rust with the ease of use of Python, providing a `statsmodels`-compatible API with significant performance improvements.

## Why RustyStats?

| Feature | RustyStats | Statsmodels |
|---------|------------|-------------|
| **Parallel IRLS Solver** | ✅ Multi-threaded via Rayon | ❌ Single-threaded |
| **Native Polars Support** | ✅ Formula API with Polars | ❌ Pandas only |
| **Built-in Lasso/Elastic Net** | ✅ All GLM families | ⚠️ Limited |
| **Performance (678K rows)** | ~1 second | ~5-10 seconds |

## Quick Example
it_g
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm(
    formula="ClaimCount ~ VehPower + VehAge + C(Area) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

print(result.summary())
print(result.relativities())  # exp(coef) for pricing
```

## Documentation Structure

This documentation is organized for maintainers who may be new to Rust and/or GLMs:

### For Understanding the Math
- [**GLM Theory**](theory/glm-intro.md) - Complete mathematical foundation
- [**Distribution Families**](theory/families.md) - Variance functions and when to use each
- [**Link Functions**](theory/links.md) - Connecting linear predictors to means
- [**IRLS Algorithm**](theory/irls.md) - How GLMs are actually fitted

### For Understanding the Code
- [**Architecture Overview**](architecture/overview.md) - How components connect
- [**Rust Core Library**](architecture/rust-core.md) - The computational engine
- [**Python Bindings**](architecture/python-bindings.md) - PyO3 bridge layer

### For Maintaining the Code
- [**Rust Best Practices**](maintenance/rust-best-practices.md) - Code style and patterns
- [**Adding New Components**](maintenance/adding-family.md) - Extending the library
- [**Testing Strategy**](maintenance/testing.md) - Test organization and practices

## Key Features

### Distribution Families
- **Gaussian** - Continuous data (linear regression)
- **Poisson** - Count data (claim frequency)
- **Binomial** - Binary outcomes (logistic regression)
- **Gamma** - Positive continuous (claim severity)
- **Tweedie** - Mixed zeros and positives (pure premium)
- **QuasiPoisson/QuasiBinomial** - Overdispersed data
- **Negative Binomial** - Alternative for overdispersed counts

### Advanced Features
- **Regularization** - Ridge, Lasso, Elastic Net with cross-validation
- **Splines** - B-splines and natural splines for non-linear effects
- **Target Encoding** - CatBoost-style encoding for high-cardinality categoricals
- **Robust Standard Errors** - HC0, HC1, HC2, HC3 sandwich estimators
- **Model Diagnostics** - Calibration, discrimination, residual analysis

## Installation

```bash
# Development installation
cd rustystats
uv run maturin develop

# Run tests
uv run pytest tests/python/ -v
```

## Project Structure

```
rustystats/
├── crates/
│   ├── rustystats-core/        # Pure Rust GLM library (no Python deps)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Crate entry, re-exports
│   │       ├── error.rs        # Error types
│   │       ├── families/       # Distribution families (Gaussian, Poisson, etc.)
│   │       ├── links/          # Link functions (Identity, Log, Logit)
│   │       ├── solvers/        # IRLS, coordinate descent
│   │       ├── inference/      # Standard errors, p-values, robust SEs
│   │       ├── diagnostics/    # Residuals, calibration, discrimination
│   │       ├── splines/        # B-splines, natural splines
│   │       ├── formula/        # Formula parsing
│   │       ├── design_matrix/  # Design matrix construction
│   │       ├── regularization/ # Lasso, Ridge, Elastic Net
│   │       ├── target_encoding/# CatBoost-style encoding
│   │       └── interactions/   # Interaction term handling
│   │
│   └── rustystats/             # Python bindings (PyO3)
│       ├── Cargo.toml
│       └── src/lib.rs          # PyO3 wrappers, NumPy conversion
│
├── python/rustystats/          # Python package
│   ├── __init__.py             # Public API exports
│   ├── glm.py                  # Summary formatting functions
│   ├── formula.py              # Formula API, glm()
│   ├── families.py             # Python family wrappers
│   ├── links.py                # Python link wrappers
│   ├── splines.py              # bs(), ns() functions
│   ├── diagnostics.py          # ModelDiagnostics, explore_data()
│   ├── interactions.py         # Interaction utilities
│   └── target_encoding.py      # TargetEncoder class
│
├── tests/
│   └── python/                 # Python integration tests
│       ├── test_glm.py
│       ├── test_families.py
│       └── ...
│
├── docs/                       # MkDocs documentation (you are here)
├── examples/                   # Jupyter notebook examples
├── Cargo.toml                  # Workspace configuration
├── pyproject.toml              # Python build config (maturin)
└── mkdocs.yml                  # Documentation config
```
