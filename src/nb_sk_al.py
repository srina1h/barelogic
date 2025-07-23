import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score
import argparse
import os
import random


def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    label_col = [c for c in df.columns if c.endswith('!')][0]
    X = df.drop(label_col, axis=1).values
    y = df[label_col].values
    # Encode labels if needed
    if y.dtype == object:
        y = np.array([1 if v in ('yes', 1, '1', 'Y', 'true', 'True') else 0 for v in y])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def active_learning(X, y, n_pos=8, repeats=10, batch_size=1000, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    results = []
    pos_label = 1
    neg_label = 0
    pos_indices = np.where(y == pos_label)[0]
    neg_indices = np.where(y == neg_label)[0]
    for rep in range(repeats):
        print(f"[Repeat {rep+1}/{repeats}] Starting active learning run...")
        # Initial labeled set
        if len(pos_indices) < n_pos or len(neg_indices) < n_pos * 4:
            continue
        selected_pos = np.random.choice(pos_indices, n_pos, replace=False)
        selected_neg = np.random.choice(neg_indices, n_pos * 4, replace=False)
        labeled = list(selected_pos) + list(selected_neg)
        pool = [i for i in range(len(y)) if i not in labeled]
        step_metrics = []
        # Initial evaluation
        print(f"  [Repeat {rep+1}] Initial evaluation on all data with {len(labeled)} labeled, {len(pool)} in pool.")
        model = GaussianNB()
        model.fit(X[labeled], y[labeled])
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        step_metrics.append((precision, recall))
        acq = 0
        no_iterations = len(pool)
        while pool:
            q = 1 - (acq / no_iterations) if no_iterations > 0 else 0
            probs = model.predict_proba(X[pool])
            pool_scores = []
            for prob in probs:
                best = np.max(prob)
                rest = np.min(prob)
                numerator = best + q * rest
                denominator = abs(q * best - rest) if abs(q * best - rest) > 1e-12 else 1e-12
                pool_scores.append(numerator / denominator)
            batch_to_acquire = min(batch_size, len(pool))
            acq_indices = np.argsort(pool_scores)[-batch_to_acquire:][::-1]
            new_indices = [pool[i] for i in acq_indices]
            labeled.extend(new_indices)
            pool = [i for i in pool if i not in new_indices]
            model.fit(X[labeled], y[labeled])
            acq += batch_to_acquire
            print(f"    [Repeat {rep+1}] Acquired {acq}/{no_iterations} samples. Labeled: {len(labeled)}. Pool left: {len(pool)}.")
            y_pred = model.predict(X)
            print(f"    [Repeat {rep+1}] Evaluation after acquisition.")
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            step_metrics.append((precision, recall))
        print(f"  [Repeat {rep+1}] Final evaluation with all samples acquired.")
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        step_metrics.append((precision, recall))
        results.append(step_metrics)
    return results

def aggregate_and_save(results, out_csv):
    # results: list of [ (precision, recall), ... ] for each repeat
    if not results:
        print("No results to save.")
        return
    n_steps = max(len(run) for run in results)
    step_precisions = [[] for _ in range(n_steps)]
    step_recalls = [[] for _ in range(n_steps)]
    for run in results:
        for i, (prec, rec) in enumerate(run):
            step_precisions[i].append(prec)
            step_recalls[i].append(rec)
    def get_quartiles(lst):
        arr = np.array(lst)
        q1 = np.percentile(arr, 25) if len(arr) else 0
        median = np.percentile(arr, 50) if len(arr) else 0
        q3 = np.percentile(arr, 75) if len(arr) else 0
        return q1, median, q3
    with open(out_csv, 'w') as f:
        f.write('step,recall_Q1,recall_median,recall_Q3,precision_Q1,precision_median,precision_Q3\n')
        for i in range(n_steps):
            rq1, rmed, rq3 = get_quartiles(step_recalls[i])
            pq1, pmed, pq3 = get_quartiles(step_precisions[i])
            f.write(f'{i},{rq1:.4f},{rmed:.4f},{rq3:.4f},{pq1:.4f},{pmed:.4f},{pq3:.4f}\n')
    print(f"Results saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV file with features and label! column')
    parser.add_argument('--n_pos', type=int, default=8, help='Number of positive samples to start with')
    parser.add_argument('--repeats', type=int, default=20, help='Number of repeats')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for acquisition')
    parser.add_argument('--output', default='results_nb_sk_al.csv', help='Output CSV for results')
    args = parser.parse_args()
    X, y = load_and_normalize(args.input)
    results = active_learning(X, y, n_pos=args.n_pos, repeats=args.repeats, batch_size=args.batch_size)
    aggregate_and_save(results, args.output)

if __name__ == '__main__':
    main() 