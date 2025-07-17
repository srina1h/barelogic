import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the results
RESULTS_DIR = os.path.dirname(__file__)

# Datasets and methods
DATASETS = ["Hall", "Kitchenham", "Radjenovic", "Wahono"]
PREPS = ["minimally_processed", "preprocessed"]
METHODS = ["bl", "sklearn"]

# File pattern
def get_file(prep, dataset, method):
    return f"{RESULTS_DIR}/results_{prep}_{dataset}_num_{method}.csv"

def load_results():
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for prep in PREPS:
            for method in METHODS:
                file_path = get_file(prep, dataset, method)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Store as dict: {treatment: {'recall': [Q1, median, Q3], 'precision': [Q1, median, Q3]}}
                    treatment = f"{prep}-{method}"
                    results[dataset][treatment] = {
                        'recall': [df.iloc[0]["recall_Q1"], df.iloc[0]["recall_median"], df.iloc[0]["recall_Q3"]],
                        'precision': [df.iloc[0]["precision_Q1"], df.iloc[0]["precision_median"], df.iloc[0]["precision_Q3"]],
                    }
    return results

def plot_side_by_side_boxplots_with_median_bar(results):
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        for ax_idx, metric in enumerate(["recall", "precision"]):
            data = []
            labels = []
            medians = []
            for treatment, vals in results.get(dataset, {}).items():
                data.append(vals[metric])
                labels.append(treatment)
                medians.append(vals[metric][1])
            if data:
                box = axes[ax_idx].boxplot(data, labels=labels, showmeans=True, patch_artist=True)
                x = range(1, len(labels) + 1)
                axes[ax_idx].bar(x, medians, width=0.4, color='lightblue', alpha=0.5, label='Median (bar)')
                axes[ax_idx].set_ylabel(metric.title())
                axes[ax_idx].set_title(f"{dataset} - {metric.title()} (Q1, Median, Q3 as box plot, median as bar)")
                axes[ax_idx].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"boxplot_medianbar_sidebyside_{dataset}.png"))
        plt.close()

def main():
    results = load_results()
    plot_side_by_side_boxplots_with_median_bar(results)

if __name__ == "__main__":
    main()
