import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label!'].value_counts()}")
    
    # Separate features and target
    X = df.drop('label!', axis=1)
    if df['label!'].dtype == 'object':
        y = df['label!'].map({'no': 0, 'yes': 1})
    else:
        y = df['label!']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution after encoding:\n{pd.Series(y).value_counts()}")
    
    # Apply min-max normalization
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y

def perform_cross_validation(X, y, n_splits=5, n_runs=100):
    # Initialize lists to store results
    all_recall_scores = []
    run_means = []
    
    # Perform multiple runs of cross-validation
    for run in range(n_runs):
        if (run + 1) % 10 == 0:  # Print progress every 10 runs
            print(f"\nCompleted {run + 1} runs...")
        
        # Initialize KFold with different random state for each run
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run)
        
        # Initialize lists to store recall scores for this run
        run_recall_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = GaussianNB()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate recall
            recall = recall_score(y_val, y_pred)
            run_recall_scores.append(recall)
        
        # Store results for this run
        all_recall_scores.extend(run_recall_scores)
        run_means.append(np.mean(run_recall_scores))
    
    # Calculate overall statistics
    overall_mean = np.mean(all_recall_scores)
    overall_std = np.std(all_recall_scores)
    
    print(f"\nOverall Statistics:")
    print(f"Mean Recall: {overall_mean:.4f}")
    print(f"Standard Deviation: {overall_std:.4f}")
    print(f"95% Confidence Interval: [{overall_mean - 1.96*overall_std:.4f}, {overall_mean + 1.96*overall_std:.4f}]")
    
    return all_recall_scores, run_means

def plot_results(all_recall_scores, run_means, experiment_name):
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot 1: Distribution of all recall scores
    sns.histplot(all_recall_scores, kde=True, ax=ax1)
    ax1.set_title('Distribution of All Recall Scores')
    ax1.set_xlabel('Recall Score')
    ax1.set_ylabel('Frequency')
    
    # Plot 2: Box plot of recall scores by fold
    sns.boxplot(data=np.array(all_recall_scores).reshape(-1, 5), ax=ax2)
    ax2.set_title('Recall Scores by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Recall Score')
    
    # Plot 3: Distribution of mean recall scores across runs
    sns.histplot(run_means, kde=True, ax=ax3)
    ax3.set_title('Distribution of Mean Recall Scores\nAcross 100 Runs')
    ax3.set_xlabel('Mean Recall Score')
    ax3.set_ylabel('Frequency')
    
    # Add experiment name as suptitle
    plt.suptitle(f'Naive Bayes Classification Results - {experiment_name}\n(100 Runs, 5-Fold CV)', y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{experiment_name.replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_balanced_sampling_experiment(X, y, n_positive_samples=100, n_runs=100):
    """
    Perform experiment with balanced sampling:
    - Randomly select n positive samples
    - Randomly select 4n negative samples
    - Train on selected samples, test on remaining data
    - Repeat 100 times and calculate statistics
    """
    recall_scores = []
    precision_scores = []
    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1} runs...")
        # Get indices of positive and negative samples
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        # Randomly sample positive and negative samples
        selected_pos = np.random.choice(pos_indices, size=n_positive_samples, replace=False)
        selected_neg = np.random.choice(neg_indices, size=n_positive_samples * 4, replace=False)
        # Combine selected samples for training
        train_indices = np.concatenate([selected_pos, selected_neg])
        test_indices = np.setdiff1d(np.arange(len(y)), train_indices)
        # Split data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)
        # Make predictions and calculate recall
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall_scores.append(recall)
        precision_scores.append(precision)
        print(f"Recall: {recall:.3f}", flush=True)
        print(f"Precision: {precision:.3f}", flush=True)
    # Calculate statistics
    median_recall = np.median(recall_scores)
    q25_recall = np.percentile(recall_scores, 25)
    q75_recall = np.percentile(recall_scores, 75)
    print("\nBalanced Sampling Experiment Results:")
    print(f"Median Recall: {median_recall:.4f}")
    print(f"25th Percentile Recall: {q25_recall:.4f}")
    print(f"75th Percentile Recall: {q75_recall:.4f}")
    return recall_scores

def main():
    print("[DEBUG] nb_sk.py is starting...")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Naive Bayes Classification with Cross Validation')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--experiment', required=True, help='Name of the experiment')
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"[DEBUG] nb_sk.py parsed arguments: experiment={args.experiment}, input={args.input}")
    print(f"Running experiment: {args.experiment}")
    print(f"Input file: {args.input}")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(args.input)
    
    # Count positive samples
    n_positive_available = np.sum(y == 1)
    print(f"\nNumber of positive samples in dataset: {n_positive_available}")
    
    # Define desired n_positive values
    desired_n_positive = [32]
    
    # Filter n_positive values to only include those less than available positive samples
    n_positive_values = [n for n in desired_n_positive if n <= n_positive_available]
    
    if not n_positive_values:
        print(f"\nError: Dataset has only {n_positive_available} positive samples, which is less than the minimum desired value of {min(desired_n_positive)}")
        return
    
    print(f"\nRunning balanced sampling experiments for n_positive values: {n_positive_values}")
    print("=" * 80)
    
    # Initialize results dictionary
    results = {
        'n_positive': n_positive_values,
        '50th_percentile': [],
        '25th_percentile': [],
        '75th_percentile': []
    }
    
    for n_pos in n_positive_values:
        print(f"\nResults for n_positive = {n_pos}:")
        print("-" * 40)
        balanced_recall_scores = perform_balanced_sampling_experiment(X, y, n_positive_samples=n_pos)
        
        # Store results
        results['50th_percentile'].append(np.median(balanced_recall_scores))
        results['25th_percentile'].append(np.percentile(balanced_recall_scores, 25))
        results['75th_percentile'].append(np.percentile(balanced_recall_scores, 75))
        print("-" * 40)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('n_positive')
    results_df = results_df.T  # Transpose to get desired format
    
    # Save to CSV without index
    output_filename = f"{args.experiment.replace(' ', '_')}_results.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()
