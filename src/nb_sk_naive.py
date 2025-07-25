import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
import argparse


def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    label_col = [c for c in df.columns if c.endswith('!')][0]
    X = df.drop(label_col, axis=1).values
    y = df[label_col].values
    # Encode labels if needed
    if y.dtype == object:
        y = np.array([1 if str(v).lower() in ('yes', '1', 'y', 'true', 'positive') else 0 for v in y])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def compute_metrics(y_true, y_pred, class_values):
    results = {v: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for v in class_values}
    for actual, predicted in zip(y_true, y_pred):
        for v in class_values:
            if actual == v and predicted == v:
                results[v]['tp'] += 1
            elif actual == v and predicted != v:
                results[v]['fn'] += 1
            elif actual != v and predicted == v:
                results[v]['fp'] += 1
            elif actual != v and predicted != v:
                results[v]['tn'] += 1
    return results

def safe_div(a, b):
    return a / b if b else 0

def print_metrics_table(results, class_values):
    print(f"{'Label':>10} {'TN':>6} {'FN':>6} {'FP':>6} {'TP':>6} {'PD':>8} {'Prec':>8} {'PF':>8} {'Acc':>8}")
    print(f"{'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    total = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for v in class_values:
        tp = results[v]['tp']
        fp = results[v]['fp']
        tn = results[v]['tn']
        fn = results[v]['fn']
        total['tp'] += tp
        total['fp'] += fp
        total['tn'] += tn
        total['fn'] += fn
        pd = safe_div(tp, tp + fn)  # recall
        prec = safe_div(tp, tp + fp)
        pf = safe_div(fp, fp + tn)
        acc = safe_div(tp + tn, tp + tn + fp + fn)
        print(f"{str(v):>10} {tn:6} {fn:6} {fp:6} {tp:6} {pd:8.3f} {prec:8.3f} {pf:8.3f} {acc:8.3f}")
    # Overall
    tp = total['tp']
    fp = total['fp']
    tn = total['tn']
    fn = total['fn']
    pd = safe_div(tp, tp + fn)
    prec = safe_div(tp, tp + fp)
    pf = safe_div(fp, fp + tn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    print(f"{'ALL':>10} {tn:6} {fn:6} {fp:6} {tp:6} {pd:8.3f} {prec:8.3f} {pf:8.3f} {acc:8.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV file with features and label! column')
    args = parser.parse_args()
    X, y = load_and_normalize(args.input)
    class_values = sorted(list(set(y)))
    model = GaussianNB()
    model.fit(X, y)
    y_pred = model.predict(X)
    results = compute_metrics(y, y_pred, class_values)
    print_metrics_table(results, class_values)

if __name__ == '__main__':
    main() 