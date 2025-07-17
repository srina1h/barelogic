# BareLogic Naive Bayes: Ablation Testing Strategy

## Overview
This document outlines the ablation testing strategy for incrementally applying enhancements to the BareLogic Naive Bayes implementation. Each change is categorized, dependencies are noted, and a recommended order for testing is provided to facilitate a systematic ablation study.

---

## Change Categories & Acronyms

### Independent Changes (Can be applied separately)

- **PAR**: Parameter Adjustments
  - Change of `k`, `m`, `BIG` values; addition of `var_smoothing_gnb`, `alpha_cnb`.
- **NST**: Numerical Stability Improvements
  - Use of `float('inf')`, `BIG_EPS`, and minimum probability thresholds.
- **TCH**: Type Checking
  - Validation of numerical values in `add()` and error raising for type mismatches.
- **EH**: Error Handling
  - Checks for empty lists, division by zero, and missing values.
- **ALM**: Active Learning Modification
  - Simplified row removal in active learning loop.

### Dependent Changes (Must be grouped)

- **DSE**: Data Structure Enhancements
  - Adds `global_num_categories` to `Sym`, `all_rows_for_global_stats` to `Data`, and normalizer propagation in `clone()`.
- **GCS**: Global Categorical Statistics _(depends on DSE)_
  - Computes and uses global category counts for categorical smoothing.
- **WAL**: Welford's Algorithm _(depends on PAR)_
  - Implements Welford's algorithm for online mean/variance and variance smoothing.
- **NOR**: Normalization Support _(depends on DSE)_
  - Adds min-max normalization and normalizer propagation.
- **NBR**: Naive Bayes Rewrite _(depends on GCS, WAL, PAR)_
  - Complete rewrite of likelihood calculation, including categorical and Gaussian smoothing, log-space calculations, and improved missing value handling.

---

## Dependency Graph

- **DSE** → **GCS**, **NOR**
- **PAR** → **WAL**, **NBR**
- **GCS**, **WAL** → **NBR**

---

## Recommended Ablation Study Order

### Phase 1: Foundation Changes (Independent)
1. **PAR** - Parameter Adjustments
2. **NST** - Numerical Stability Improvements
3. **TCH** - Type Checking
4. **EH** - Error Handling
5. **ALM** - Active Learning Modification

### Phase 2: Infrastructure Changes
6. **DSE** - Data Structure Enhancements

### Phase 3: Feature-Specific Changes
7. **GCS** - Global Categorical Statistics (requires DSE)
8. **WAL** - Welford's Algorithm (requires PAR)
9. **NOR** - Normalization (requires DSE)

### Phase 4: Core Algorithm Changes
10. **NBR** - Naive Bayes Rewrite (requires GCS, WAL, PAR)

---

## Testing Strategy

For each phase:
1. Apply the change(s) in the recommended order.
2. Run your performance and accuracy tests.
3. Compare results with the previous version.
4. Document the impact on accuracy, speed, and memory usage.

This approach will help you:
- Identify which changes have the most impact on performance and accuracy.
- Determine the necessity of each change for your use case.
- Find the optimal subset of changes for your requirements. 