# Functional Changes in `src/bl.py`

## 1. Option Defaults and Parameters
- Changed default values for `-k` and `-m` (low frequency Bayes hack) from 1/2 to 0, aligning with empirical priors.
- Added new options for Naive Bayes smoothing:
  - `-v var_smoothing_gnb`: Variance smoothing for GaussianNB (default: 1e-9).
  - `-V alpha_cnb`: Alpha for categorical (Laplace) smoothing (default: 1.0).
  - `-x BIG_EPS`: Small constant for numerical stability (default: 1e-30).

## 2. Data Structures and Initialization
- `Num` and `Sym` objects now use more robust initialization for min/max and category tracking.
- `Sym` columns now track `global_num_categories` for correct Laplace smoothing across all classes.
- `Data` objects now collect all rows in `all_rows_for_global_stats` to enable global statistics for categorical features.

## 3. Add/Subtract Logic for Data and Columns
- `add` and `sub` functions for `Num` and `Sym` columns now:
  - Properly update counts and statistics for both addition and subtraction.
  - Use Welford's algorithm for online mean/variance, with special handling for subtraction and small sample sizes.
  - For `Sym`, update `n` only for non-missing values.
  - For `Data`, row count (`n`) is updated only for non-missing rows.
- Added robust handling for missing values and type mismatches in `add`.

## 4. Normalization and Statistics
- `norm` now avoids division by zero and returns 0.5 if the range is zero.
- `spread` (entropy/stddev) now guards against division by zero for symbolic columns.
- `mid` and `yNums` now handle empty or missing data gracefully.

## 5. Naive Bayes Model (likes/like)
- `likes` now updates `global_num_categories` for all `Sym` columns before prediction, ensuring correct Laplace smoothing.
- `like`:
  - For symbolic features, uses CategoricalNB-style smoothing: `(count + alpha) / (total + alpha * n_categories)`.
  - For numeric features, uses Gaussian likelihood with variance smoothing.
  - Class prior is now empirical: `(data.n + k) / (nall + k*nh + BIG_EPS)`.
  - All log-probabilities are stabilized with `BIG_EPS` to avoid math errors.

## 6. Active Learning and Tree Functions
- `actLearn` and related functions now avoid problematic subtraction logic and instead only add to datasets, simplifying model updates.
- All tree and cut logic now robustly handle missing values and edge cases.

## 7. Utility and Helper Functions
- `first` now returns an empty string for empty lists.
- `coerce` is more robust to whitespace and boolean string values.
- `csv` uses multiline regex for robust line cleaning.
- `adds` and `clone` now ensure correct initialization and statistics for all data types.

## 8. Miscellaneous
- All calculations involving division or logarithms now use `BIG_EPS` to avoid division by zero or log(0).
- All random operations are seeded with `the.rseed` for reproducibility.

---