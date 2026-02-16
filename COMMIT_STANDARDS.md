# Commit Standards

Every commit merged into `main` must satisfy the checks below. Review this list during code review and before approving a PR.

---

## Design Philosophy

These are the non-negotiable principles that shape every decision in rustystats. If a commit conflicts with any of these, it needs rethinking.

### Numerical correctness first

Results must be verifiable against established references (statsmodels, R). Prefer well-understood algorithms (IRLS, coordinate descent, PAVA) over novel approaches. Every solver change must include numerical comparison tests.

### Rust computes, Python orchestrates

Hot loops, linear algebra, and data-parallel work live in `crates/rustystats-core/`. Python (`python/rustystats/`) owns the user-facing API, formula parsing, and result presentation. PyO3 bindings in `crates/rustystats/` are a thin bridge — no business logic.

### Minimal dependencies

Core runtime requires only `numpy` and `polars`. No scipy, no pandas, no sklearn in `[project.dependencies]`. Heavy packages belong in `[project.optional-dependencies]` or `[dependency-groups]` only.

### Dict-first API

The primary interface is `glm_dict()` — a single dict describing the model. This is designed for programmatic construction by agents and automated workflows, not just human typing. Keep the dict schema stable and well-documented.

### Performance without sacrificing clarity

Use Rayon for parallelism, cache intermediate results, avoid unnecessary allocations — but not at the cost of readable code. Profile before optimising. Benchmark claims must be reproducible from `benchmarks/`.

### One code path

Penalized and unpenalized fits, all families, all term types flow through the same IRLS/solver pipeline. No special-case branches per family unless mathematically required. Parallel implementations drift and break.

### Fail loud, fail early

Invalid inputs (NaN, rank-deficient matrices, impossible constraints) must produce clear error messages, not silent garbage. The `validate()` method exists so users can check before fitting — but fitting itself must also reject bad inputs.

---

## Engineering Standards

## 1. No Duplication (DRY)

- No copy-pasted logic. If two places do the same thing, extract a shared function/module.
- Shared Rust utilities live in `crates/rustystats-core/src/`. Python utilities live in `python/rustystats/`.
- If the same logic exists in both Rust and Python (e.g. spline basis construction), the Rust version is canonical. The Python version should call into Rust or be clearly marked as a fallback.

## 2. Simplicity (KISS)

- Prefer the simplest solution that works. No premature abstractions.
- If a function takes more than 5 parameters, consider a config object or breaking it up.
- Avoid clever one-liners that sacrifice readability.

## 3. Single Responsibility

- Each module, class, and function does one thing.
- PyO3 bindings (`crates/rustystats/src/`) are thin — type conversion and delegation only. Algorithm logic lives in `rustystats-core`.
- Python API functions (`python/rustystats/`) own user-facing orchestration: formula parsing, result formatting, diagnostics aggregation. No linear algebra in Python.

## 4. Type Safety

- **Python**: All function signatures have type annotations. No `Any` unless truly unavoidable.
- **Rust**: Use strong types. Prefer `newtype` wrappers over raw `f64`/`usize` where semantics matter. No `unwrap()` in library code — propagate errors with `Result` and `?`.

## 5. Linter Clean

- `ruff check python/` must pass with zero errors before merge.
- `cargo clippy --workspace` must pass with no warnings.
- New ruff or clippy rules are not silenced without a comment explaining why.

## 6. No Dead Code

- No unused imports, variables, or functions.
- No commented-out code blocks. Use version control to retrieve old code.
- No empty files. If a file has no content, delete it.

## 7. No Stale Documentation

- If you change behaviour, update the relevant doc in `docs/`.
- Status tables in design docs must reflect reality (no "❌ Missing" for things that exist).
- README examples must actually run.

## 8. Dependency Discipline

- No heavy optional dependencies in core `[project.dependencies]`. Use `[project.optional-dependencies]`.
- Pin minimum versions, not exact versions.
- Every new dependency must be justified in the PR description.

## 9. No Resource Leaks

- **Python**: File handles use `with` statements. Never `open()` without close.
- **Rust**: No `unsafe` blocks without a `// SAFETY:` comment justifying correctness. Prefer safe abstractions from `ndarray`/`nalgebra`.

## 10. Build Reproducibility

- `maturin develop --release` must produce a working package from a clean checkout.
- All Rust tests pass: `cargo test --workspace`.
- All Python tests pass: `pytest tests/python/`.
- CI must replicate local results. If it works locally but not in CI, the commit is not ready.

## 11. Correct Data Structures

- Use `deque` for FIFO, not `list.pop(0)`.
- Use `set` for membership checks, not list scans.
- Prefer `ndarray::Array2` for dense matrices in Rust. Use column-major layout when interfacing with LAPACK/numpy.
- Accept `polars.DataFrame` at the API boundary; convert to numpy arrays for computation.

## 12. Error Handling

- Never swallow exceptions silently (`except: pass`).
- **Rust**: Return `Result<T, E>` with descriptive error variants. Use `thiserror` or `anyhow` crate conventions. Panics in library code are bugs.
- **Python**: Raise specific exceptions (`ValueError`, `LinAlgError`) with messages that help the user fix the problem. The `validate()` method should catch issues before they become cryptic solver failures.

## 13. Consistent Naming

- **Python**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Rust**: `snake_case` for functions/variables, `PascalCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- PyO3-exposed function names match the Python convention (`snake_case`), not the Rust convention for the same function.

## 14. Numerical Hygiene

- Guard against division by zero, log of zero, and overflow in link/variance functions.
- Use step-halving or other safeguards when IRLS produces non-finite deviance.
- Spline knot placement must handle edge cases (duplicate values, values outside knot range).
- Tolerance constants (convergence, rank detection) must be named constants, not magic numbers.

## 15. Security Basics

- No hardcoded secrets or API keys. Use environment variables.
- No `eval()` on untrusted input. Expression terms (`expr`) are parsed, not evaluated raw.
- Sensitive internal docs (competitive analysis, credentials) must be in `.gitignore`.

## 16. Test Coverage

- New business logic must include at least one test.
- Bug fixes must include a regression test.
- Tests must not depend on external services or network access.
- Never delete or weaken an existing test without explicit justification in the PR.

## 17. Commit Hygiene

- Each commit is atomic: one logical change per commit.
- Commit messages follow conventional format: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
- No generated files (`__pycache__/`, `target/`, build output) in commits.

---

## LLM-Generated Code: Watch For These

AI coding assistants produce plausible-looking code that often hides real problems. Every reviewer (human or AI) must check for these patterns specifically.

### Dangerous fallbacks that mask errors

```python
# BAD — silently returns empty data instead of crashing
def get_data(path):
    try:
        return pl.read_parquet(path)
    except Exception:
        return pl.DataFrame()  # caller has no idea it failed
```

- Never return a default value from a `catch` unless the caller explicitly expects it.
- Prefer letting exceptions propagate. If you must catch, log the error and re-raise or return a result type that signals failure.
- Watch for `or {}`, `or []`, `or ""` fallbacks that turn a bug into silent wrong data.

### Broad exception swallowing

- `except Exception: pass` is almost never correct. Catch the specific exception you expect.
- Broad catches are acceptable only at the outermost public API boundary (e.g. `glm_dict().fit()`). Inner code should not catch broadly.

### Hallucinated APIs and parameters

- LLMs invent function signatures, config keys, and library methods that don't exist. Every API call, import, and parameter must be verified against the actual codebase or library docs.
- Watch for plausible-but-wrong Polars methods (e.g. `df.groupby()` instead of `df.group_by()`).

### Stale patterns from older library versions

- LLMs train on old code. Watch for deprecated patterns:
  - `from typing import List, Dict` → use `list`, `dict` (Python 3.10+)
  - `pd.DataFrame` when we use `pl.DataFrame`
  - `np.matrix` or `scipy.linalg` when we use `ndarray`/`nalgebra` in Rust
  - Old Polars APIs (e.g. `apply` → `map_elements`, `groupby` → `group_by`)

### Defensive code that hides bugs

```python
# BAD — if term_config is missing a key, this silently uses wrong defaults
df = term_config.get("df", {}).get("value", 3)
```

- Chained `.get()` with default dicts makes `KeyError` impossible to diagnose. If the key should exist, access it directly and let it fail loudly.
- Only use `.get()` with defaults when the key is genuinely optional.

### Duplicated logic disguised as "safety"

- LLMs often generate a "just in case" check that duplicates logic already handled upstream. This creates two code paths that must be kept in sync.
- If a function already validates its input, don't re-validate it in the caller.

### Over-abstraction and premature generalisation

- LLMs love creating `BaseSolver`, `SolverFactory`, `AbstractFamilyDispatcher` hierarchies for code that has exactly one concrete implementation.
- If there's only one subclass, you don't need the base class.
- Prefer plain functions over class hierarchies until you have three concrete use cases.

### Numerically plausible but wrong implementations

- LLMs generate statistical code that *looks* correct but has subtle numerical issues: wrong derivative signs, missing weight terms, incorrect deviance formulas.
- Every new family, link function, or solver change must be validated against statsmodels or R output on a known dataset.
- Watch for missing edge-case guards (log of zero, division by zero variance) that only surface on real data.

### Comments that restate the code

```python
# BAD — the comment adds zero information
x = x + 1  # increment x by 1
```

- LLMs pad output with obvious comments. Comments should explain *why*, not *what*.
- Delete any comment that a competent reader could infer from the code itself.

### Untested edge cases presented as handled

- LLMs generate `if` branches for edge cases but don't test them. An untested branch is worse than no branch — it gives false confidence.
- If you add an edge case handler, add a test for it. If you can't test it, add a `# TODO: untested` comment.

### Import bloat

- LLMs import modules speculatively. If a function isn't used, the import shouldn't be there.
- Watch for `from typing import ...` lines that grow with every edit but never shrink.

---

## Backward Compatibility: None Required

Rustystats is pre-1.0. No backward compatibility guarantees exist yet.

- **No compatibility shims** — if an API is poorly designed, change it. Do not keep the old version alongside the new one.
- **No deprecation warnings** — if something is wrong, remove it. Do not add `warnings.warn` with a future removal date.
- **No serialization versioning** — `to_bytes()` / `from_bytes()` format can change freely. Users re-fit models.
- **No feature flags** — if a feature is ready, ship it. Do not add `ENABLE_NEW_X` environment variables.

Prioritize clean, simple code over compatibility gymnastics.

---

## Quick Checklist

Copy into PR descriptions:

```
Design Philosophy
- [ ] Numerical results verified against reference (statsmodels/R)
- [ ] Computation in Rust, orchestration in Python, PyO3 bindings are thin
- [ ] No new core runtime dependencies (only numpy + polars)
- [ ] Dict API schema unchanged (or migration path documented)
- [ ] Single code path — no special-case branches per family unless mathematically required
- [ ] Invalid inputs produce clear errors, not silent wrong results

Engineering Standards
- [ ] No duplicated logic
- [ ] All Python functions have type annotations
- [ ] No `unwrap()` in Rust library code — use `Result` + `?`
- [ ] `ruff check python/` passes with zero errors
- [ ] `cargo clippy --workspace` passes with no warnings
- [ ] No unused imports, variables, or dead code
- [ ] Docs updated if behaviour changed
- [ ] No new heavy dependencies in core
- [ ] `maturin develop --release` builds cleanly
- [ ] `cargo test --workspace` and `pytest tests/python/` pass
- [ ] Numerical edge cases guarded (div/0, log(0), overflow)
- [ ] Tests added for new logic; bug fixes include regression tests

LLM Code Review
- [ ] No silent fallbacks that mask errors (return empty data, `or {}`, `or []`)
- [ ] No broad exception swallowing (`except Exception: pass`)
- [ ] All API calls and imports verified against actual codebase/library docs
- [ ] No deprecated patterns (old typing imports, pandas, old Polars APIs)
- [ ] No chained .get() on keys that should exist — fail loudly on missing data
- [ ] No redundant validation that duplicates upstream checks
- [ ] No premature abstraction (base classes with one subclass)
- [ ] Comments explain why, not what — no restating the code
- [ ] Edge case branches have tests, or are marked # TODO: untested
- [ ] Statistical code validated against known reference output

Backward Compatibility
- [ ] No compatibility shims, version flags, or migration code
- [ ] Bad APIs are replaced, not versioned alongside
```
