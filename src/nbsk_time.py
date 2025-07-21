import sys
import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

if len(sys.argv) < 2:
    print("Usage: python nbsk_time.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
n_train = 5000
n_test = 3000
n_repeats = 100

# Load data
all_data = pd.read_csv(csv_file)
all_rows = all_data.values
class_col_idx = -1  # assume last column is class

accs = []
fit_times = []
predict_times = []

for _ in range(n_repeats):
    np.random.shuffle(all_rows)
    train_rows = all_rows[:n_train]
    test_rows = all_rows[n_train:n_train+n_test]
    X_train = train_rows[:, :class_col_idx]
    y_train = train_rows[:, class_col_idx]
    X_test = test_rows[:, :class_col_idx]
    y_test = test_rows[:, class_col_idx]

    clf = GaussianNB()
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    t2 = time.perf_counter()

    acc = np.mean(y_pred == y_test)
    accs.append(acc)
    fit_times.append((t1 - t0) * 1000)      # ms
    predict_times.append((t2 - t1) * 1000)  # ms

mu = np.mean(accs)
var = np.var(accs, ddof=1)
sd = np.std(accs, ddof=1)
mean_fit = np.mean(fit_times)
mean_pred = np.mean(predict_times)

print(f"sklearn GaussianNB: mean={mu:.4f}, var={var:.6f}, sd={sd:.6f}, n={n_repeats}")
print(f"Average fit time: {mean_fit:.2f} ms, Average inference time: {mean_pred:.2f} ms") 