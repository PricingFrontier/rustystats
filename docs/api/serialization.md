# Model Serialization

Save and load fitted models for production deployment and later use.

## Quick Start

```python
import rustystats as rs

# Fit and save
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "bs"},
        "Region": {"type": "categorical"},
        "Brand": {"type": "target_encoding"},
    },
    data=train_data,
    family="poisson",
    exposure="Exposure",
).fit()

# Serialize to bytes
model_bytes = result.to_bytes()

# Save to file
with open("model.bin", "wb") as f:
    f.write(model_bytes)
```

## Loading Models

```python
# Load from file
with open("model.bin", "rb") as f:
    loaded = rs.GLMModel.from_bytes(f.read())

# Predict with loaded model
predictions = loaded.predict(new_data)
```

Native multinomial models use the same pattern:

```python
result = rs.multinomial_dict(
    response="PurchasedTier",
    terms={"DriverAge": {"type": "linear"}, "Channel": {"type": "categorical"}},
    data=quotes,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
).fit()

model_bytes = result.to_bytes()
loaded = rs.MultinomialModel.from_bytes(model_bytes)
probabilities = loaded.predict_proba(new_quotes)
```

Models fitted with deterministic `input_transforms` are also self-contained.
The serialized payload stores the canonical transform specs and recompiles
lookup tables when loaded, so production callers can pass the same raw columns
to `loaded.predict(...)` that they passed to the original model.

---

## API Reference

### to_bytes()

Serialize fitted model to bytes.

```python
model_bytes = result.to_bytes()
```

**Returns:** `bytes` - Binary representation of the model.

### GLMModel.from_bytes()

Load a model from bytes.

```python
loaded = rs.GLMModel.from_bytes(model_bytes)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | bytes | Serialized model bytes |

**Returns:** `GLMModel` - Loaded model ready for prediction.

### MultinomialModel.from_bytes()

Load a native multinomial model from bytes.

```python
loaded = rs.MultinomialModel.from_bytes(model_bytes)
```

**Returns:** `MultinomialModel` - Loaded model ready for `predict_proba`,
`predict`, `predict_top_k`, `tier_mix`, diagnostics on supplied data, and
scenario scoring when the original model used `alternative_terms`.

---

## What's Preserved

The serialized model includes everything needed for prediction:

| Component | Included | Notes |
|-----------|----------|-------|
| Coefficients | ✓ | All fitted parameters |
| Feature names | ✓ | For matching new data columns |
| Family & Link | ✓ | For inverse link transform |
| Categorical levels | ✓ | For encoding new data |
| Spline knots | ✓ | For basis function evaluation |
| Target encoding stats | ✓ | Prior, level means, counts |
| Frequency encoding stats | ✓ | Level frequencies |
| Input transforms | ✓ | Deterministic lookup specs for raw-data scoring |
| Formula/term specs | ✓ | For design matrix construction |

For `MultinomialModel`, serialization also preserves class order, reference
class, the coefficient matrix, alternative-term specifications and
coefficients, class availability and offset specs when they were column/scalar
based, the weight-column spec, regularization metadata, solver status, warnings,
and inference-status labels.

If multinomial availability or offset was supplied as an in-memory array at fit
time, that array is treated as training-row data and is not serialized. Loaded
models require a fresh `availability=` or `offset=` override for prediction in
that case.

## What's NOT Preserved

Training-only artifacts are excluded to minimize size:

| Component | Included | Reason |
|-----------|----------|--------|
| Training data | ✗ | Too large, not needed |
| Covariance matrix | ✗ | Can recompute if needed |
| Residuals | ✗ | Training-specific |
| Diagnostics | ✗ | Training-specific |

Standalone calibration objects are separate deployment artifacts. Serialize a
`MultinomialInterceptCalibration` with `to_dict()` / `to_json()` and restore it
with `MultinomialInterceptCalibration.from_dict(...)`; then pass it as
`calibration=` when scoring.

---

## Storage Size

Models are compact—typically kilobytes, not megabytes:

| Model Complexity | Approximate Size |
|------------------|------------------|
| 10 features, no splines | ~2 KB |
| 50 features, 3 splines | ~10 KB |
| 100 features, 10 splines, 5 TE columns | ~50 KB |

---

## Production Deployment

### Example: Flask API

```python
from flask import Flask, request, jsonify
import rustystats as rs
import polars as pl

app = Flask(__name__)

# Load model at startup
with open("model.bin", "rb") as f:
    model = rs.GLMModel.from_bytes(f.read())

@app.route("/predict", methods=["POST"])
def predict():
    data = pl.DataFrame(request.json)
    predictions = model.predict(data)
    return jsonify({"predictions": predictions.tolist()})
```

### Example: AWS Lambda

```python
import rustystats as rs
import polars as pl
import json

# Load model from S3 or bundled file
with open("model.bin", "rb") as f:
    MODEL = rs.GLMModel.from_bytes(f.read())

def handler(event, context):
    data = pl.DataFrame(json.loads(event["body"]))
    predictions = MODEL.predict(data)
    return {
        "statusCode": 200,
        "body": json.dumps({"predictions": predictions.tolist()})
    }
```

---

## Version Compatibility

The serialized state carries a `schema_version`. `GLMModel.from_bytes` requires
it to match the schema version of the running RustyStats build **exactly** and
raises a clear `ValidationError` otherwise — it does not migrate or best-effort
load older or newer payloads. (A silently mis-loaded model can mispredict, for
example when the exposure layout changed between versions, so loading fails loud
instead.)

RustyStats is pre-1.0: the serialization format may change between releases, so
**persisted models are not guaranteed to load across versions**. Re-fit and
re-serialize models after upgrading RustyStats.

**Best practice:** store each model alongside the exact RustyStats version that
wrote it (in metadata or the filename), and regenerate models when you upgrade.

---

## Troubleshooting

### Model won't load

```python
try:
    model = rs.GLMModel.from_bytes(data)
except ValueError as e:
    print(f"Failed to load: {e}")
    # Check if data is corrupted or from incompatible version
```

### Prediction fails on new data

```python
# Check that new data has required columns
required = model.required_columns
missing = set(required) - set(new_data.columns)
if missing:
    print(f"Missing columns: {missing}")
```

`required_columns` returns raw input columns. For transformed models it includes
the transform source columns, not the derived transform output columns.

### Unseen categorical levels

```python
# Unseen levels in categoricals → reference level (coefficient = 0)
# Unseen levels in target encoding → global prior
predictions = model.predict(new_data)  # Works, uses fallbacks
```
