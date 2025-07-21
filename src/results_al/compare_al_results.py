import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Usage: python compare_al_results.py <dataset_name> <start>

def get_file_path(mode, start, proc, dataset):
    return f"results_uncertainty_{mode}_{start}_{proc}_{dataset}_num.csv"

def read_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

def compare_and_plot_datasets(start, datasets):
    modes = ['bl', 'sklearn']
    procs = ['minimally_processed', 'preprocessed']
    results = {}
    for mode in modes:
        for proc in procs:
            key = f"{mode}_{proc}"
            results[key] = {}
            for dataset in datasets:
                file_path = get_file_path(mode, start, proc, dataset)
                df = read_metrics(file_path)
                if df is not None:
                    results[key][dataset] = df
                else:
                    print(f"Warning: {file_path} not found.")
    # Save images in ./img inside the same folder as this script
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'img'))
    os.makedirs(img_dir, exist_ok=True)
    # Individual plots for each dataset and each result type
    for key in results:
        for dataset in datasets:
            df = results[key].get(dataset)
            if df is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(df['step'], df['recall_median'], label=f"{dataset}")
                plt.xlabel('Step')
                plt.ylabel('Recall (Median)')
                plt.title(f'Recall (Median) vs Step - {key} - {dataset}')
                plt.legend()
                plt.tight_layout()
                recall_plot_path = os.path.join(img_dir, f'recall_median_vs_step_{key}_start{start}_{dataset}.png')
                plt.savefig(recall_plot_path)
                plt.close()
    # Combined recall plot for all 4 result types for a single dataset
    if len(datasets) == 1:
        dataset = datasets[0]
        plt.figure(figsize=(10, 6))
        for key in results:
            df = results[key].get(dataset)
            if df is not None:
                plt.plot(df['step'], df['recall_median'], label=key)
        plt.xlabel('Step')
        plt.ylabel('Recall (Median)')
        plt.title(f'Combined Recall (Median) vs Step - {dataset}')
        plt.legend(title='Mode/Processing')
        plt.tight_layout()
        combined_recall_plot_path = os.path.join(img_dir, f'combined_recall_median_vs_step_{dataset}_start{start}.png')
        plt.savefig(combined_recall_plot_path)
        plt.close()
    # Individual precision plots
    for key in results:
        for dataset in datasets:
            df = results[key].get(dataset)
            if df is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(df['step'], df['precision_median'], label=f"{dataset}")
                plt.xlabel('Step')
                plt.ylabel('Precision (Median)')
                plt.title(f'Precision (Median) vs Step - {key} - {dataset}')
                plt.legend()
                plt.tight_layout()
                precision_plot_path = os.path.join(img_dir, f'precision_median_vs_step_{key}_start{start}_{dataset}.png')
                plt.savefig(precision_plot_path)
                plt.close()
    print(f"Plots saved in {img_dir}")

def main():
    # Usage: python compare_al_results.py <start> [<dataset_name>]
    if len(sys.argv) == 2:
        start = sys.argv[1]
        datasets = ['Hall', 'Kitchenham', 'Wahono', 'Radjenovic']
        compare_and_plot_datasets(start, datasets)
    elif len(sys.argv) == 3:
        start = sys.argv[1]
        dataset = sys.argv[2]
        compare_and_plot_datasets(start, [dataset])
    else:
        print("Usage: python compare_al_results.py <start> [<dataset_name>]")
        print("If <dataset_name> is omitted, all datasets are processed.")
        print("Example: python compare_al_results.py 32 Wahono")
        sys.exit(1)

if __name__ == "__main__":
    main()
