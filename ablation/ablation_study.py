import subprocess
import re
import csv
import os
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

# Define the scripts and datasets
METHODS = [
    ("bl-before-change.py", "before_change"),
    ("bl_PAR.py", "PAR"),
    ("bl_NBR.py", "NBR"),
    ("bl_NOR.py", "NOR"),
    ("nb_sk.py", "SK"),
]

DATASETS = [
    ("preprocessed_Hall.csv", "Hall"),
    ("preprocessed_Kitchenham.csv", "Kitchenham"),
    ("preprocessed_Wahono.csv", "Wahono"),
    ("preprocessed_Radjenovic.csv", "Radjenovic"),
]

# Helper to parse recall, precision from output
RECALL_RE = re.compile(r"Recall: ([0-9.]+)")
PRECISION_RE = re.compile(r"Precision: ([0-9.]+)")

# Run eg__nbfew for a given script and dataset, parse metrics
def run_experiment(script, dataset):
    if script == "nb_sk.py":
        cmd = ["python3", "-u", os.path.join("src", script), "--input", os.path.join(".", dataset), "--experiment", "nbfew"]
    else:
        cmd = ["python3", os.path.join("src", script), "--nbfew", os.path.join(".", dataset)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout
    stderr_output = proc.stderr
    recalls = [float(x) for x in re.findall(RECALL_RE, output)]
    precisions = [float(x) for x in re.findall(PRECISION_RE, output)]
    # If precision not found, fill with None
    if not precisions:
        precisions = [None] * len(recalls)
    return recalls, precisions, output, stderr_output

# Aggregate metrics
results = []

for script, method in METHODS:
    for dataset_file, dataset_name in DATASETS:
        print(f"Running {method} on {dataset_name}...")
        recalls, precisions, output, stderr_output = run_experiment(script, dataset_file)
        if method == "SK":
            print(f"[DEBUG] SK {dataset_name}: {len(recalls)} recall, {len(precisions)} precision values parsed.")
            print(f"[DEBUG] SK {dataset_name} raw output (first 500 chars):\n{output[:500]}")
            print(f"[DEBUG] SK {dataset_name} stderr (first 500 chars):\n{stderr_output[:500]}")
        # Compute means and stds, ignoring None
        recall_mean = mean(recalls) if recalls else None
        recall_std = stdev(recalls) if len(recalls) > 1 else 0
        precisions_non_none = [p for p in precisions if p is not None]
        if precisions_non_none:
            precision_mean = mean(precisions_non_none)
            precision_std = stdev(precisions_non_none) if len(precisions_non_none) > 1 else 0
        else:
            precision_mean = precision_std = None
        results.append({
            "Method": method,
            "Dataset": dataset_name,
            "Recall Mean": recall_mean,
            "Recall Std": recall_std,
            "Precision Mean": precision_mean,
            "Precision Std": precision_std,
        })

# Write summary CSV
summary_csv = "ablation_summary.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Print summary table
print("\nAblation Study Summary:")
print(f"{'Method':<15} {'Dataset':<15} {'Recall':<20} {'Precision':<20}")
for row in results:
    recall_str = f"{row['Recall Mean']:.3f} ± {row['Recall Std']:.3f}" if row['Recall Mean'] is not None else 'N/A'
    precision_str = f"{row['Precision Mean']:.3f} ± {row['Precision Std']:.3f}" if row['Precision Mean'] is not None else 'N/A'
    print(f"{row['Method']:<15} {row['Dataset']:<15} {recall_str:<20} {precision_str:<20}")

# --- Plotting ---
def plot_all_metrics():
    method_order = ["before_change", "PAR", "NBR", "NOR", "SK"]
    method_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#FFA500"]  # orange for SK
    missing_color = "#CCCCCC"  # gray for missing
    for dataset_file, dataset_name in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        metrics = [
            ("Recall Mean", "Recall Std", "Recall"),
            ("Precision Mean", "Precision Std", "Precision"),
        ]
        for i, (metric, std_metric, ylabel) in enumerate(metrics):
            means = []
            stds = []
            labels = []
            bar_colors = []
            for j, method in enumerate(method_order):
                found = False
                for row in results:
                    if row["Method"] == method and row["Dataset"] == dataset_name:
                        if row[metric] is not None:
                            means.append(row[metric])
                            stds.append(row[std_metric])
                            bar_colors.append(method_colors[j])
                        else:
                            means.append(0)
                            stds.append(0)
                            bar_colors.append(missing_color)
                        found = True
                        break
                if not found:
                    means.append(0)
                    stds.append(0)
                    bar_colors.append(missing_color)
                labels.append(method)
            x = np.arange(len(labels))
            axes[i].bar(x, means, yerr=stds, capsize=8, color=bar_colors)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(labels)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f"{ylabel}")
        plt.suptitle(f"Ablation Study Comparison on {dataset_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        outname = f"ablation_{dataset_name}.png"
        plt.savefig(outname)
        print(f"Saved plot: {outname}")
        plt.close()

plot_all_metrics() 