import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Usage: python compare_diabetes.py <start>

def get_file_path(model, start):
    if model == 'bl':
        return f"results_uncertainty_bl_{start}_diabetes.csv"
    elif model == 'bl_new':
        return f"results_uncertainty_sklearn_{start}_diabetes.csv"
    elif model == 'sklearn':
        return f"results_uncertainty_gaussian_nb_{start}_diabetes.csv"
    else:
        return None

def read_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    recall_cols = ['recall_Q1', 'recall_median', 'recall_Q3']
    precision_cols = ['precision_Q1', 'precision_median', 'precision_Q3']
    all_zero = (df[recall_cols + precision_cols] == 0).all(axis=1)
    df = df[~all_zero]
    return df

def compare_and_plot_diabetes(start):
    sns.set(style="whitegrid", context="talk", palette="tab10")
    model_map = {'bl': 'BL', 'bl_new': 'BL (sklearn)', 'sklearn': 'GaussianNB'}
    models = ['bl', 'bl_new', 'sklearn']
    results = {}
    for model in models:
        file_path = get_file_path(model, start)
        df = read_metrics(file_path)
        if df is not None:
            results[model] = df
        else:
            print(f"Warning: {file_path} not found.")
    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'img'))
    os.makedirs(img_dir, exist_ok=True)
    # Recall plot
    plt.figure(figsize=(12, 7))
    found = False
    import itertools
    color_palette = sns.color_palette('tab10', n_colors=3)
    color_cycle = itertools.cycle(color_palette)
    for model in models:
        df = results.get(model)
        if df is not None:
            color = next(color_cycle)
            plt.plot(df['step'], df['recall_median'], label=model_map[model], color=color, linewidth=2)
            plt.fill_between(df['step'], df['recall_Q1'], df['recall_Q3'], color=color, alpha=0.2)
            found = True
    if found:
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Recall (Median)', fontsize=14)
        plt.title(f'Recall (Median) vs Step - diabetes (all models, start={start})', fontsize=16, pad=20)
        plt.legend(title='Model', fontsize=12, title_fontsize=13, loc='lower right', frameon=True)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        note = (
            "Model key: BL = Original BL, BL (sklearn) = BL with sklearn implementation, GaussianNB = sklearn GaussianNB.\n"
            "Shaded area: Interquartile range (Q1-Q3) of recall."
        )
        plt.gcf().text(0.5, -0.12, note, ha='center', va='top', fontsize=10, wrap=True, transform=plt.gca().transAxes)
        recall_plot_path = os.path.join(img_dir, f'recall_median_vs_step_ALLMODELS_start{start}_diabetes.png')
        plt.savefig(recall_plot_path, bbox_inches='tight')
        plt.close()
    # False alarm rate plot
    plt.figure(figsize=(12, 7))
    found = False
    color_cycle = itertools.cycle(color_palette)
    for model in models:
        df = results.get(model)
        if df is not None and 'false_alarm_median' in df.columns:
            color = next(color_cycle)
            plt.plot(df['step'], df['false_alarm_median'], label=model_map[model], color=color, linewidth=2)
            plt.fill_between(df['step'], df['false_alarm_Q1'], df['false_alarm_Q3'], color=color, alpha=0.2)
            found = True
    if found:
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('False Alarm Rate (%) (Median)', fontsize=14)
        plt.title(f'False Alarm Rate (Median) vs Step - diabetes (all models, start={start})', fontsize=16, pad=20)
        plt.legend(title='Model', fontsize=12, title_fontsize=13, loc='upper right', frameon=True)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        note = (
            "Model key: BL = Original BL, BL (sklearn) = BL with sklearn implementation, GaussianNB = sklearn GaussianNB.\n"
            "Shaded area: Interquartile range (Q1-Q3) of false alarm rate."
        )
        plt.gcf().text(0.5, -0.12, note, ha='center', va='top', fontsize=10, wrap=True, transform=plt.gca().transAxes)
        fa_plot_path = os.path.join(img_dir, f'false_alarm_median_vs_step_ALLMODELS_start{start}_diabetes.png')
        plt.savefig(fa_plot_path, bbox_inches='tight')
        plt.close()
    print(f"Diabetes recall and false alarm plots (all models) saved in {img_dir}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_diabetes.py <start>")
        sys.exit(1)
    start = sys.argv[1]
    compare_and_plot_diabetes(start)

if __name__ == "__main__":
    main() 