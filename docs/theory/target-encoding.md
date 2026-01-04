# Target Encoding: The CatBoost Approach

This chapter explains how RustyStats handles high-cardinality categorical variables using CatBoost-style ordered target encoding. We'll build the intuition step-by-step, show why naive approaches fail, and explain why the CatBoost method works.

---

## The Problem: High-Cardinality Categoricals

### What's a High-Cardinality Categorical?

A categorical variable with many unique values:

| Variable | Example Values | Cardinality |
|----------|----------------|-------------|
| Gender | Male, Female, Other | Low (3) |
| US State | CA, TX, NY, ... | Medium (50) |
| ZIP Code | 90210, 10001, ... | High (40,000+) |
| Brand | Toyota, Ford, ... | High (varies) |
| Customer ID | C001, C002, ... | Very High (millions) |

### Why Are They Problematic?

**One-hot encoding** creates a column for each category:

```
Brand_Toyota  Brand_Ford  Brand_Honda  ...  (10,000 columns!)
    1             0           0
    0             1           0
    0             0           1
```

Problems:
- **Dimensionality explosion**: 10,000 brands = 10,000 new columns
- **Sparse data**: Most columns are 0
- **Rare categories**: Some brands appear only once—can't estimate a reliable coefficient
- **Overfitting**: Model memorizes rare categories instead of learning patterns

---

## Naive Solution: Mean Target Encoding

### The Idea

Replace each category with the mean of the target for that category:

```
Category    Target    →    Encoded
   A          1              0.67   (mean of A's: 1,1,0)
   B          0              0.33   (mean of B's: 1,0,0)
   A          1              0.67
   B          1              0.33
   A          0              0.67
   B          0              0.33
```

This seems clever—we compress arbitrary categories into a single informative number!

### The Fatal Flaw: Target Leakage

**Target leakage** occurs when information about the target "leaks" into the features.

Consider a rare category that appears only once:

```
Category    Target    →    Encoded
   RARE        1              1.0   (mean of just this one observation)
```

The encoding for "RARE" is **exactly its target value**! The model can achieve perfect prediction on training data by memorizing these encodings—but it will fail on new data.

**This is overfitting in disguise.**

Even for categories with a few observations, the encoding contains substantial information about the specific target values in those rows.

---

## The CatBoost Solution: Ordered Target Statistics

### The Key Insight

**Don't let an observation see its own target** (or targets from "future" observations).

CatBoost's solution:
1. Randomly order all observations
2. For each observation, compute the encoding using **only observations that came before it** in the random order

### Step-by-Step Example

Data (6 observations):
```
Index   Category   Target
  0        A         1
  1        B         0  
  2        A         1
  3        B         0
  4        A         0
  5        B         1
```

**Step 1**: Random permutation order: [3, 0, 5, 2, 4, 1]

**Step 2**: Process in order, tracking running statistics:

| Process Order | Original Index | Category | Target | Sum_before | Count_before | Encoded |
|---------------|----------------|----------|--------|------------|--------------|---------|
| 1st | 3 | B | 0 | 0 | 0 | **prior** |
| 2nd | 0 | A | 1 | 0 | 0 | **prior** |
| 3rd | 5 | B | 1 | 0 | 1 | 0/1 = 0.0 |
| 4th | 2 | A | 1 | 1 | 1 | 1/1 = 1.0 |
| 5th | 4 | A | 0 | 2 | 2 | 2/2 = 1.0 |
| 6th | 1 | B | 0 | 1 | 2 | 1/2 = 0.5 |

The first observation of each category **sees no data** for that category—it gets the prior (global mean).

### Why This Prevents Leakage

- Each observation is encoded using **only data it hasn't "seen"**
- An observation's own target is **never used** in its encoding
- Rare categories get the prior (global mean) as their first encoding—no memorization!

---

## Adding Regularization: The Prior

### The Problem with Raw Statistics

What if a category has only 1 observation before the current one?

```
Encoded = 1/1 = 1.0   (or 0/1 = 0.0)
```

This is very noisy! One observation doesn't give reliable information.

### The Solution: Blend with Prior

Add a "pseudo-count" that pulls the estimate toward the global mean:

$$
\text{Encoded} = \frac{\text{sum\_target} + \text{prior} \times \text{prior\_weight}}{\text{count} + \text{prior\_weight}}
$$

Where:
- **prior** = global mean of target (e.g., 0.5)
- **prior_weight** = regularization strength (e.g., 1.0)

### Example with prior_weight = 1.0

If prior = 0.5 and we've seen 1 observation with target = 1:

$$
\text{Encoded} = \frac{1.0 + 0.5 \times 1.0}{1 + 1.0} = \frac{1.5}{2} = 0.75
$$

Instead of 1.0, we get 0.75—pulled toward the global mean.

### Interpretation of Prior Weight

| prior_weight | Effect |
|--------------|--------|
| 0 | No regularization (raw statistics) |
| 1 | Balance: 1 observation ≈ 1 "prior observation" |
| 10 | Heavy regularization: need 10+ observations to overcome prior |
| 100 | Very heavy: rare categories essentially get the prior |

**Rule of thumb**: Start with prior_weight = 1.0. Increase if overfitting on rare categories.

---

## Multiple Permutations: Reducing Variance

### The Problem

The encoding depends on the random permutation order:

- Permutation [0,1,2,3,4,5] gives different encodings than [5,4,3,2,1,0]
- This adds noise to the features

### The Solution

Average encodings across multiple random permutations:

```python
encoded = average([
    encode_with_permutation(seed=1),
    encode_with_permutation(seed=2),
    encode_with_permutation(seed=3),
    encode_with_permutation(seed=4),
])
```

### How Many Permutations?

| n_permutations | Variance | Speed |
|----------------|----------|-------|
| 1 | High | Fast |
| 4 (default) | Medium | Good balance |
| 10 | Low | Slower |
| 100 | Very low | Much slower |

**Rule of thumb**: 4 permutations is usually sufficient. Increase if you see high variance in cross-validation.

---

## The Complete Algorithm

```
Algorithm: CatBoost-style Ordered Target Encoding

Input:
  - categories[n]: categorical values
  - target[n]: target variable
  - prior_weight: regularization strength (default: 1.0)
  - n_permutations: number of random orderings (default: 4)

Output:
  - encoded[n]: encoded values

1. Compute prior = mean(target)

2. For each permutation p = 1 to n_permutations:
   
   a. Generate random permutation order π
   
   b. Initialize: sum_by_category = {}, count_by_category = {}
   
   c. For i in permutation order π:
      category = categories[i]
      sum_before = sum_by_category.get(category, 0)
      count_before = count_by_category.get(category, 0)
      
      encoded_p[i] = (sum_before + prior × prior_weight) / (count_before + prior_weight)
      
      sum_by_category[category] += target[i]
      count_by_category[category] += 1

3. Return average across permutations: encoded = mean(encoded_1, ..., encoded_p)
```

---

## Applying to New Data

For **prediction on new data**, we don't need ordering—we use the full training statistics:

$$
\text{Encoded}_{\text{new}} = \frac{\text{sum\_target\_train} + \text{prior} \times \text{prior\_weight}}{\text{count\_train} + \text{prior\_weight}}
$$

For **unseen categories** (not in training), use the prior:

$$
\text{Encoded}_{\text{unseen}} = \text{prior}
$$

---

## RustyStats Implementation

### Basic Usage

```python
import rustystats as rs
import numpy as np

# Training data
categories = ["Toyota", "Ford", "Toyota", "Honda", "Ford", "Toyota"]
target = np.array([1.0, 0.0, 1.0, 0.5, 0.0, 0.8])

# Encode
encoded, name, prior, stats = rs.target_encode(
    categories, 
    target, 
    var_name="brand",
    prior_weight=1.0,
    n_permutations=4,
    seed=42  # For reproducibility
)

print(f"Encoded values: {encoded}")
print(f"Column name: {name}")  # "TE(brand)"
print(f"Prior (global mean): {prior:.3f}")
```

### For Prediction

```python
# New data (including unseen category)
new_categories = ["Toyota", "Ford", "BMW"]  # BMW not in training

# Apply encoding using training statistics
new_encoded = rs.apply_target_encoding(
    new_categories, 
    stats,      # From training
    prior,      # From training
    prior_weight=1.0
)

print(new_encoded)
# Toyota: uses training stats
# Ford: uses training stats  
# BMW: gets prior (unseen category)
```

### Scikit-learn Style API

```python
# Fit-transform pattern
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4, seed=42)

# Training: uses ordered statistics
train_encoded = encoder.fit_transform(train_categories, train_target)

# Prediction: uses full training statistics
test_encoded = encoder.transform(test_categories)
```

### In Formulas

```python
# TE() in formulas automatically applies target encoding
result = rs.glm(
    "claims ~ TE(brand) + TE(region) + age + C(gender)",
    data,
    family="poisson"
).fit()
```

---

## When to Use Target Encoding

### Good Use Cases

| Scenario | Why Target Encoding Helps |
|----------|---------------------------|
| High-cardinality categoricals (100+ levels) | Avoids dimensionality explosion |
| Rare categories | Prior regularization prevents overfitting |
| Tree-unfriendly models (GLMs) | Compresses categories to single feature |
| Memory constraints | One column instead of thousands |

### When NOT to Use

| Scenario | Better Alternative |
|----------|-------------------|
| Low-cardinality (< 10 levels) | One-hot encoding or C() |
| Very few observations | Just use prior (or drop variable) |
| Categories have no relationship to target | Drop the variable |
| Interpretability is critical | One-hot (coefficients are clearer) |

---

## Comparison with Alternatives

| Method | Handles High Cardinality | Prevents Leakage | Regularization | Interpretable |
|--------|--------------------------|------------------|----------------|---------------|
| One-hot encoding | ❌ | ✅ | ❌ | ✅ |
| Label encoding | ✅ | ✅ | ❌ | ❌ |
| Naive mean encoding | ✅ | ❌ | ❌ | ⚠️ |
| **CatBoost target encoding** | ✅ | ✅ | ✅ | ⚠️ |
| Leave-one-out encoding | ✅ | ⚠️ | ❌ | ⚠️ |

**CatBoost-style is the only method that handles all three challenges**: high cardinality, target leakage, and rare categories.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Target leakage** | When features contain information about the target they shouldn't have |
| **Ordered statistics** | Encode each observation using only "past" data in random order |
| **Prior** | Global mean used for regularization and unseen categories |
| **Prior weight** | Controls regularization strength (higher = more pull toward prior) |
| **Multiple permutations** | Average across random orderings to reduce variance |

**The CatBoost approach is elegant**: by simply changing the order of computation and adding a prior, we solve both overfitting and high-cardinality problems in one technique.

---

## References

- Prokhorenkova et al. (2018). [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516). NeurIPS.
- Micci-Barreca (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD.

---

## Next Steps

- [GLM Workflow Guide](../guides/glm-workflow.md) — Using target encoding in practice
- [Formula API](../api/formula-api.md) — TE() syntax in formulas
- [Code Walkthrough](../rust-guide/code-walkthrough.md) — Implementation details
