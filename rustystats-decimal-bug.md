# Bug Report: `explore_data` crashes on Decimal columns with nulls

## Summary

`rs.explore_data()` raises a `TypeError` when the input DataFrame contains columns with `Decimal` dtype that have null values. The error occurs in `discretize()` when numpy tries to sort an array containing `decimal.Decimal` values mixed with `None`.

## Environment

- Python 3.13
- rustystats 0.4.11
- numpy (latest)
- Data source: Parquet file read via Polars (which preserves Decimal types)

## Reproduction

Any Polars DataFrame with a `Decimal` column containing at least one null will trigger the crash when passed to `explore_data`:

```python
import polars as pl
import rustystats as rs

df = pl.DataFrame({
    "response": [1, 0, 1, 0, 1],
    "group": ["a", "a", "b", "b", "a"],
    "premium": pl.Series([10.50, 20.75, None, 15.00, 30.25]).cast(pl.Decimal),
})

rs.explore_data(
    data=df,
    response="response",
    family="binomial",
    categorical_factors=["group"],
    continuous_factors=["premium"],
)
```

## Error

```
TypeError: '<' not supported between instances of 'decimal.Decimal' and 'NoneType'
```

## Traceback

```
explore_data (explorer.py:1234)
  → detect_interactions (explorer.py:962)
    → _discretize (explorer.py:1002)
      → discretize (utils.py:37)
        → np.unique(values)
          → ar.sort()  ← fails here
```

`np.unique()` internally calls `.sort()` on the array. When the array contains `decimal.Decimal` objects mixed with `None` (from Polars null), Python's `<` operator is undefined between the two types.

## Suggested Fix

In `diagnostics/utils.py`, the `discretize` function should cast input to `float64` and handle NaN before sorting:

```python
def discretize(values, n_bins):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    unique_vals = np.unique(values)
    # ... rest of function
```

This handles both issues:
1. **Decimal → float64 conversion**: numpy can sort native floats
2. **Null/None → NaN filtering**: avoids comparison errors with missing values

## Impact

When `explore_data` fails, downstream consumers (e.g. Atelier) lose:
- All exploration chart data (univariate/bivariate plots)
- The null model fit (and therefore Rao's score test for factor ranking)
- Interaction detection candidates

## Workaround

Callers can cast Decimal columns to Float64 before passing data to rustystats:

```python
decimal_cols = [c for c in df.columns if df[c].dtype.base_type() == pl.Decimal]
if decimal_cols:
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in decimal_cols])
```
