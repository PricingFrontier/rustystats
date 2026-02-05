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
    offset="Exposure",
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
| Formula/term specs | ✓ | For design matrix construction |

## What's NOT Preserved

Training-only artifacts are excluded to minimize size:

| Component | Included | Reason |
|-----------|----------|--------|
| Training data | ✗ | Too large, not needed |
| Covariance matrix | ✗ | Can recompute if needed |
| Residuals | ✗ | Training-specific |
| Diagnostics | ✗ | Training-specific |

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

Models are forward-compatible within the same major version:

| Saved With | Loadable By |
|------------|-------------|
| v1.0.x | v1.0.x, v1.1.x, v1.2.x |
| v1.1.x | v1.1.x, v1.2.x |
| v2.x.x | v2.x.x only |

**Best practice:** Include the RustyStats version in your model metadata or filename.

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
required = model.feature_names
missing = set(required) - set(new_data.columns)
if missing:
    print(f"Missing columns: {missing}")
```

### Unseen categorical levels

```python
# Unseen levels in categoricals → reference level (coefficient = 0)
# Unseen levels in target encoding → global prior
predictions = model.predict(new_data)  # Works, uses fallbacks
```
