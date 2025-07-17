# Implementation Changes: BareLogic Naive Bayes Enhancement

## Overview

This document details the computational differences between the original implementation (`bl-before-change.py`) and the enhanced version (`bl.py`). The primary focus is on improving the naive Bayes classifier to match scikit-learn's implementation while maintaining numerical stability and adding new features.

## Key Configuration Changes

### Parameter Updates
- **`k`**: Changed from `1` to `0` (low frequency Bayes hack)
- **`m`**: Changed from `2` to `0` (low frequency Bayes hack)
- **`BIG`**: Changed from `1E32` to `1e-30` (renamed to `BIG_EPS`)
- **New parameters**:
  - `var_smoothing_gnb`: `1e-9` (variance smoothing for GaussianNB)
  - `alpha_cnb`: `1.0` (alpha for categorical smoothing)
  - `BIG_EPS`: `1e-30` (epsilon constant)

## Core Data Structure Enhancements

### Sym Object Enhancement
**Before:**
```python
def Sym(txt=" ", at=0):
    return o(it=Sym, txt=txt, at=at, n=0, has={})
```

**After:**
```python
def Sym(txt=" ", at=0):
    return o(it=Sym, txt=txt, at=at, n=0, has={}, global_num_categories=0)
```

**Impact**: Added `global_num_categories` to track unique categorical values across all classes for proper smoothing in CategoricalNB.

### Data Object Enhancement
**Before:**
```python
def Data(src=[]): 
    return adds(src, o(it=Data,n=0,rows=[],cols=None))
```

**After:**
```python
def Data(src=[]):
    data_obj = o(it=Data, n=0, rows=[], cols=None, all_rows_for_global_stats=[])
    return adds(src, data_obj)
```

**Impact**: Added `all_rows_for_global_stats` to collect all training rows for calculating global categorical statistics.

### Clone Function Enhancement
**Before:**
```python
def clone(data, src=[]): 
    return adds(src, Data([data.cols.names]))
```

**After:**
```python
def clone(data, src=[]):
    new_data = Data([data.cols.names])
    if hasattr(data, 'normalizer'):
        new_data.normalizer = data.normalizer
    return adds(src, new_data)
```

**Impact**: Added support for normalizer propagation during cloning.

## Statistical Computation Changes

### Num Object Initialization
**Before:**
```python
def Num(txt=" ", at=0):
    return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG, 
             rank=0, goal = 0 if str(txt)[-1]=="-" else 1)
```

**After:**
```python
def Num(txt=" ", at=0):
    return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, 
             hi=-float("inf"), lo=float("inf"), rank=0,
             goal=0 if str(txt)[-1] == "-" else 1)
```

**Impact**: Changed from `BIG` to `float("inf")` for better numerical stability.

### Add Function: Welford's Algorithm Implementation
**Before:**
```python
def _num():
    i.lo = min(v, i.lo)
    i.hi = max(v, i.hi)
    if flip < 0 and i.n < 2: 
        i.mu = i.sd = 0
    else:
        d = v - i.mu
        i.mu += flip * (d / i.n)
        i.m2 += flip * (d * (v - i.mu))
        i.sd = 0 if i.n <= 2 else (max(0,i.m2)/(i.n-1))**.5
```

**After:**
```python
def _num():
    if v != "?":
        if not isNum(v):
            raise TypeError("Expected numerical value for Num column")
        
        if flip > 0:  # Adding a value
            i.n += n
            if i.n == 0:
                i.mu = v
                i.m2 = 0
            else:
                delta = v - i.mu
                i.mu += delta / i.n
                i.m2 += delta * (v - i.mu)
        elif flip < 0:  # Subtracting a value
            # Complex inverse Welford implementation
            if i.n > n:
                i.n -= n
                if i.n == 0:
                    i.mu = 0
                    i.m2 = 0
                else:
                    old_mu = (i.mu * (i.n + n) - v * n) / i.n
                    delta_v_old_mu = v - old_mu
                    delta_v_new_mu = v - i.mu
                    i.m2 = i.m2 - (delta_v_old_mu * (v - i.mu))
                    i.m2 = max(0, i.m2)
            else:
                i.n = 0
                i.mu = 0
                i.sd = 0
                i.m2 = 0
                i.lo = -float("inf")
                i.hi = float("inf")
        
        # Variance calculation with smoothing
        if i.n > 1:
            variance = i.m2 / (i.n - 1)
        elif i.n == 1:
            variance = 0
        else:
            variance = 0
        
        i.sd = (variance + the.var_smoothing_gnb) ** 0.5
```

**Key Changes**:
1. **Type checking**: Added validation for numerical values
2. **Welford's algorithm**: Implemented proper online mean/variance calculation
3. **Variance smoothing**: Added `var_smoothing_gnb` to match scikit-learn's GaussianNB
4. **Sample variance**: Uses `(n-1)` denominator for sample variance
5. **Inverse Welford**: Complex implementation for subtraction operations

## Naive Bayes Implementation Overhaul

### Likes Function: Global Categorical Statistics
**Before:**
```python
def likes(lst, datas):
    n = sum(data.n for data in datas)
    return max(datas, key=lambda data: like(lst, data, n, len(datas)))
```

**After:**
```python
def likes(lst, datas):
    nall = sum(d.n for d in datas)
    nh = len(datas)
    
    # Calculate global categorical statistics
    if datas:
        all_feature_values_by_index = {}
        for d in datas:
            for row_data in d.all_rows_for_global_stats:
                if d.cols:
                    for col_idx, value in enumerate(row_data):
                        if (col_idx < len(d.cols.all) and 
                            d.cols.all[col_idx].it is Sym and value != "?"):
                            all_feature_values_by_index.setdefault(col_idx, set()).add(value)
        
        for d in datas:
            for col in d.cols.all:
                if col.it is Sym:
                    col.global_num_categories = len(
                        all_feature_values_by_index.get(col.at, set())
                    )
                    if col.global_num_categories == 0:
                        col.global_num_categories = 2
    
    return max(datas, key=lambda data: like(lst, data, nall, nh))
```

**Impact**: Pre-computes global categorical statistics for proper smoothing across all classes.

### Like Function: Complete Rewrite
**Before:**
```python
def like(row, data, nall=100, nh=2):
    def _col(v,col): 
        if col.it is Sym: 
            return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
        sd = col.sd + 1/BIG
        nom = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
        denom = (2*math.pi*sd*sd) ** 0.5
        return max(0, min(1, nom/denom))

    prior = (data.n + the.k) / (nall + the.k*nh)
    tmp = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]
    return sum(math.log(n) for n in tmp + [prior] if n>0)
```

**After:**
```python
def like(row, data, nall=100, nh=2):
    if hasattr(data, 'normalizer') and data.normalizer:
        row = data.normalizer.normalize(row)
    
    def _col(v, col):
        if v == "?":
            return 1.0

        if col.it is Sym:
            # CategoricalNB smoothing
            n_categories_for_smoothing = max(1, col.global_num_categories)
            return (col.has.get(v, 0) + the.alpha_cnb) / (
                col.n + the.alpha_cnb * n_categories_for_smoothing + the.BIG_EPS
            )

        # Gaussian likelihood
        sd = col.sd
        if sd <= the.BIG_EPS:
            return 1.0 if abs(v - col.mu) < the.BIG_EPS else the.BIG_EPS

        # Log-space calculation for numerical stability
        log_nom = -1 * (v - col.mu) ** 2 / (2 * sd * sd)
        log_denom = 0.5 * math.log(2 * math.pi * sd * sd)
        log_pdf = log_nom - log_denom
        pdf = math.exp(log_pdf)
        
        min_prob = 1e-10
        return max(pdf, min_prob)

    # Class prior calculation
    prior = (data.n + the.k) / (nall + the.k * nh + the.BIG_EPS)
    prior = max(prior, 1e-10)

    tmp = []
    for x_col in data.cols.x:
        if row[x_col.at] != "?":
            val_likelihood = _col(row[x_col.at], x_col)
            tmp.append(val_likelihood)

    log_prior = math.log(prior) if prior > the.BIG_EPS else math.log(the.BIG_EPS)
    log_likelihoods_sum = sum(math.log(max(n, the.BIG_EPS)) for n in tmp)

    return log_prior + log_likelihoods_sum
```

**Key Changes**:
1. **Normalization support**: Added optional feature normalization
2. **Categorical smoothing**: Implemented proper CategoricalNB smoothing with global category counts
3. **Gaussian smoothing**: Added variance smoothing for numerical features
4. **Numerical stability**: Log-space calculations and minimum probability thresholds
5. **Missing value handling**: Consistent treatment of missing values
6. **Prior calculation**: Improved class prior calculation with epsilon protection

## Mathematical Formulae

### Categorical Likelihood (CategoricalNB)
**Formula**: P(feature | class) = (count + α) / (total_class_count + α × n_categories)

**Implementation**:
```python
return (col.has.get(v, 0) + the.alpha_cnb) / (
    col.n + the.alpha_cnb * n_categories_for_smoothing + the.BIG_EPS
)
```

### Gaussian Likelihood (GaussianNB)
**Formula**: P(feature | class) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

**Implementation**:
```python
log_nom = -1 * (v - col.mu) ** 2 / (2 * sd * sd)
log_denom = 0.5 * math.log(2 * math.pi * sd * sd)
log_pdf = log_nom - log_denom
pdf = math.exp(log_pdf)
```

### Class Prior
**Formula**: P(class) = (class_count + k) / (total_count + k × n_classes)

**Implementation**:
```python
prior = (data.n + the.k) / (nall + the.k * nh + the.BIG_EPS)
```

## Additional Enhancements

### Normalization Support
Added `Normalizer` class for min-max normalization:
```python
class Normalizer:
    def __init__(self, cols):
        self.mins = [float('inf')] * len(cols.x)
        self.maxs = [float('-inf')] * len(cols.x)
        self.idxs = [col.at for col in cols.x]
    
    def update(self, row):
        for i, idx in enumerate(self.idxs):
            v = row[idx]
            if isNum(v):
                if v < self.mins[i]: self.mins[i] = v
                if v > self.maxs[i]: self.maxs[i] = v
    
    def normalize(self, row):
        normed = list(row)
        for i, idx in enumerate(self.idxs):
            v = row[idx]
            if isNum(v):
                lo, hi = self.mins[i], self.maxs[i]
                if hi > lo:
                    normed[idx] = (v - lo) / (hi - lo)
                else:
                    normed[idx] = 0.0
        return normed
```

### Error Handling and Validation
- Added type checking for numerical values
- Improved handling of edge cases (zero variance, missing values)
- Better numerical stability with epsilon constants

### Active Learning Modifications
**Before:**
```python
add(sub(best.rows.pop(-1), best), rest)
```

**After:**
```python
removed_row = best.rows.pop(-1)
add(removed_row, rest)
```

**Impact**: Simplified row removal to avoid complex inverse Welford operations.

## Performance Implications

1. **Memory**: Increased due to storing global categorical statistics
2. **Computation**: More complex but numerically stable calculations
3. **Accuracy**: Improved matching with scikit-learn's implementation
4. **Robustness**: Better handling of edge cases and numerical issues

## Compatibility Notes

- The enhanced version maintains backward compatibility for most functions
- New parameters have sensible defaults that match scikit-learn
- The core API remains unchanged for existing users
- Experimental functions (cross-validation, etc.) are additive and don't affect core functionality 