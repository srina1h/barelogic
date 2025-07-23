import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Usage: python compare_al_results.py <start> [<dataset_name>]

def get_file_path(model, start, proc, dataset):
    # Map to correct file patterns
    if model == 'bl':
        return f"results_uncertainty_bl_{start}_{proc}_{dataset}_num.csv"
    elif model == 'bl_new':
        return f"results_uncertainty_sklearn_{start}_{proc}_{dataset}_num.csv"
    elif model == 'sklearn':  # gaussian_nb
        return f"results_uncertainty_gaussian_nb_{start}_{dataset}_{proc}.csv"
    else:
        return None

def read_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # Remove rows where all recall and precision columns are zero
    recall_cols = ['recall_Q1', 'recall_median', 'recall_Q3']
    precision_cols = ['precision_Q1', 'precision_median', 'precision_Q3']
    all_zero = (df[recall_cols + precision_cols] == 0).all(axis=1)
    df = df[~all_zero]
    return df

def compare_and_plot_datasets(start, datasets):
    sns.set(style="whitegrid", context="talk", palette="tab10")
    # Model and processing nomenclature mapping
    model_map = {'bl': 'bl', 'bl_new': 'bl_new', 'sklearn': 'sklearn'}
    proc_map = {'minimally_processed': 'minimal', 'preprocessed': 'NLTK'}
    # All models and procs
    models = ['bl', 'bl_new', 'sklearn']
    procs = ['minimally_processed', 'preprocessed']
    results = {}
    for model in models:
        results[model] = {}
        for proc in procs:
            results[model][proc] = {}
            for dataset in datasets:
                file_path = get_file_path(model, start, proc, dataset)
                df = read_metrics(file_path)
                if df is not None:
                    results[model][proc][dataset] = df
                else:
                    print(f"Warning: {file_path} not found.")
    # Save images in ./img inside the same folder as this script
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'img'))
    os.makedirs(img_dir, exist_ok=True)
    # Only save the all models & processing recall plot for each dataset
    for dataset in datasets:
        plt.figure(figsize=(12, 7))
        found = False
        # Use a color palette for better visibility
        import itertools
        color_palette = sns.color_palette('tab10', n_colors=6)
        color_cycle = itertools.cycle(color_palette)
        for model in models:
            for proc in procs:
                df = results[model][proc].get(dataset)
                if df is not None:
                    color = next(color_cycle)
                    # Plot recall median
                    plt.plot(df['step'], df['recall_median'], label=f'{model_map[model]}-{proc_map[proc]}', color=color, linewidth=2)
                    # Shade between Q1 and Q3 for variance
                    plt.fill_between(df['step'], df['recall_Q1'], df['recall_Q3'], color=color, alpha=0.2)
                    found = True
        if found:
            plt.xlabel('Step', fontsize=14)
            plt.ylabel('Recall (Median)', fontsize=14)
            plt.title(f'Recall (Median) vs Step - {dataset} (all models & processing, start={start})', fontsize=16, pad=20)
            plt.legend(title='Model-Processing', fontsize=12, title_fontsize=13, loc='lower right', frameon=True)
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            # Add explanation note at the bottom
            note = (
                "Model key: bl = Original BL, bl_new = BL with sklearn implementation, sklearn = GaussianNB (sklearn).\n"
                "Preprocessing: minimal = minimal preprocessing, NLTK = NLTK-based preprocessing.\n"
                "Shaded area: Interquartile range (Q1-Q3) of recall."
            )
            # Place the note just below the x-axis label
            plt.gcf().text(0.5, -0.12, note, ha='center', va='top', fontsize=10, wrap=True, transform=plt.gca().transAxes)
            recall_plot_path = os.path.join(img_dir, f'recall_median_vs_step_ALLMODELS_start{start}_{dataset}_minimal_vs_NLTK.png')
            plt.savefig(recall_plot_path, bbox_inches='tight')
            plt.close()
    print(f"Recall plots (all models & processing) saved in {img_dir}")

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
