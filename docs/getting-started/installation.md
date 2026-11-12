# Installation

RustyStats requires Python 3.10 or later. `pip install rustystats` (or `uv add rustystats`)
installs a ready-built wheel on Linux (x86_64 and aarch64), Windows (x64) and macOS (Intel and
Apple silicon), so no Rust toolchain is needed. You only need Rust to build from source, as
described below.

## Prerequisites for building from source

### Python
Ensure you have Python 3.10 or later:
```bash
python --version  # Should be 3.10+
```

### Rust
Install Rust via rustup if not already installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Verify installation:
```bash
rustc --version  # Should show version 1.70+
cargo --version
```

### uv (Recommended)
We use `uv` for Python package management. Install via the official installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Development Installation

Clone and build from source:

```bash
# Clone the repository
git clone https://github.com/PricingFrontier/rustystats.git
cd rustystats

# Install in development mode
uv run maturin develop
```

!!! note "Development Mode"
    `maturin develop` compiles the Rust code and installs it in your current Python environment. Changes to Rust code require re-running this command.

## Verify Installation

```python
import rustystats as rs
import numpy as np

# Quick test
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
X = np.column_stack([np.ones(5), np.array([1, 2, 3, 4, 5])])

result = rs.fit_glm(y, X, family="gaussian")
print(f"Coefficients: {result.params}")
print(f"RustyStats is working!")
```

## Dependencies

### Core (Required)
- `numpy>=1.20` - Array operations

### Optional
- `polars>=2.0,<3` - DataFrame support for formula API
- `pyarrow` - Parquet file support

### Development
- `pytest>=7.0` - Testing
- `maturin>=1.4` - Build tool
- `statsmodels>=0.14` - Comparison testing

## Troubleshooting

### Compilation Errors

If Rust compilation fails:

1. **Update Rust**: `rustup update`
2. **Check Cargo.toml**: Ensure all dependencies are available
3. **Clean build**: `cargo clean` then rebuild

### Import Errors

If `import rustystats` fails:

1. Ensure you ran `maturin develop` (not just `maturin build`)
2. Check you're in the correct Python environment
3. Verify with `uv pip list | grep rustystats`

### Performance Issues

For optimal performance:

1. Build in release mode: `maturin develop --release`
2. Ensure Rayon parallelism isn't disabled by environment variables
