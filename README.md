# RustyStats 🦀📊

**Fast Generalized Linear Models with a Rust Backend**

A statsmodels-compatible Python library for GLM fitting, optimized for actuarial applications.

## Why RustyStats?

- **Fast**: Core algorithms implemented in Rust for speed
- **Familiar**: API designed to match statsmodels
- **Clear**: Heavily documented code that actuaries can understand and maintain
- **Reliable**: Comprehensive test suite validated against statsmodels

## Installation

### From PyPI (coming soon)
```bash
pip install rustystats
```

### From Source (Development)
```bash
# Clone the repository
git clone https://github.com/your-org/rustystats.git
cd rustystats

# Install Rust if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install in development mode
maturin develop
```

## Quick Start

```python
import rustystats as rs
import numpy as np

# Distribution families
poisson = rs.families.Poisson()
gamma = rs.families.Gamma()
binomial = rs.families.Binomial()

# Link functions
log_link = rs.links.Log()
logit_link = rs.links.Logit()

# Check variance functions
mu = np.array([1.0, 2.0, 5.0])
print(poisson.variance(mu))  # [1.0, 2.0, 5.0] - variance = mean
print(gamma.variance(mu))    # [1.0, 4.0, 25.0] - variance = mean²

# GLM fitting coming soon!
# model = rs.GLM(y, X, family=rs.families.Poisson())
# result = model.fit()
# print(result.summary())
```

## Available Families

| Family | Variance Function | Default Link | Use For |
|--------|-------------------|--------------|---------|
| Gaussian | V(μ) = 1 | Identity | Linear regression |
| Poisson | V(μ) = μ | Log | Claim frequency |
| Binomial | V(μ) = μ(1-μ) | Logit | Binary outcomes |
| Gamma | V(μ) = μ² | Log | Claim severity |

## Available Links

| Link | Function | Inverse | Use With |
|------|----------|---------|----------|
| Identity | η = μ | μ = η | Gaussian |
| Log | η = log(μ) | μ = exp(η) | Poisson, Gamma |
| Logit | η = log(μ/(1-μ)) | μ = sigmoid(η) | Binomial |

## For Actuaries

This library is designed with actuarial pricing in mind:

### Claim Frequency
```python
# Poisson GLM with log link
# Coefficients are log rate relativities
# exp(β) = rate relativity
```

### Claim Severity  
```python
# Gamma GLM with log link
# Coefficients are log severity relativities
# exp(β) = severity relativity
```

### Pure Premium
```python
# If both use log link:
# Pure premium β = Frequency β + Severity β
# (Because log(Freq × Sev) = log(Freq) + log(Sev))
```

## Project Structure

```
rustystats/
├── Cargo.toml                 # Rust workspace config
├── pyproject.toml             # Python package config
├── crates/
│   ├── rustystats-core/       # Pure Rust statistics library
│   │   └── src/
│   │       ├── families/      # Distribution families
│   │       ├── links/         # Link functions
│   │       └── error.rs       # Error types
│   └── rustystats/            # Python bindings (PyO3)
├── python/
│   └── rustystats/            # Python package
│       ├── __init__.py
│       ├── families.py
│       └── links.py
└── tests/
    ├── rust/                  # Rust tests (cargo test)
    └── python/                # Python tests (pytest)
```

## Development

### Running Tests

```bash
# Rust tests
cargo test

# Python tests (after building)
maturin develop
pytest tests/python/
```

### Building Documentation

```bash
# Rust docs
cargo doc --open

# Python docs (coming soon)
```

## Roadmap

- [x] Phase 1: Project scaffolding
- [x] Phase 2: Link functions (Identity, Log, Logit)
- [x] Phase 3: Distribution families (Gaussian, Poisson, Binomial, Gamma)
- [ ] Phase 4: IRLS fitting algorithm
- [ ] Phase 5: Statistical inference (standard errors, p-values)
- [ ] Phase 6: Model diagnostics (residuals, influence)
- [ ] Phase 7: Python API (statsmodels-compatible)
- [ ] Phase 8: Formula interface (patsy/formulaic)
- [ ] Phase 9: Advanced features (regularization, robust SE)

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Please read the code documentation and follow the existing style.
The code is intentionally verbose with lots of comments to help actuaries understand and maintain it.
