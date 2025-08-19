# Naive Bayes Active Learning Experiments

This directory contains the naive_bayes_AL.py script and a simplified Makefile that automatically runs all experiments with different n_pos values (8, 16, 32) and dynamically discovers treatment directories.

## Files

- `naive_bayes_AL.py` - The main script for running naive Bayes active learning experiments
- `naive_bayes_AL_SMOTE.py` - Version with SMOTE applied to initial sampled datasets
- `generate_makefile.py` - Script to generate a simplified Makefile that automatically runs all n_pos values
- `Makefile` - Generated Makefile with simplified targets (automatically runs all n_pos values)

## Usage

### Using the Makefile (Recommended)

The Makefile provides simplified targets that automatically run all n_pos values (8, 16, 32) for each experiment:

#### Run all regular experiments (all treatments, all datasets, all n_pos values):
```bash
make all
```

#### Run all SMOTE experiments (all treatments, all datasets, all n_pos values):
```bash
make all-smote
```

#### Run experiments for a specific treatment (all datasets, all n_pos values):
```bash
make run-treatment TREATMENT=tfidf_50_max_features_10000_top_n_50
```

#### Run SMOTE experiments for a specific treatment (all datasets, all n_pos values):
```bash
make run-treatment-smote TREATMENT=tfidf_50_max_features_10000_top_n_50
```

#### Run experiments for a specific dataset (all treatments, all n_pos values):
```bash
make run-dataset DATASET=Hall
```

#### Run SMOTE experiments for a specific dataset (all treatments, all n_pos values):
```bash
make run-dataset-smote DATASET=Hall
```

#### Check running experiments:
```bash
make status
```

#### Clean up log files:
```bash
make clean
```

#### Show help:
```bash
make help
```

### Using the Python script

```bash
python3 run_all_experiments.py
```

## Directory Structure

Results will be saved in the following structure:
```
../results/
├── treatment_name_1/
│   ├── Hall/
│   │   ├── results_nb_sk_al_n8.csv
│   │   ├── results_nb_sk_al_n16.csv
│   │   ├── results_nb_sk_al_n32.csv
│   │   ├── experiment_n8.log
│   │   ├── experiment_n16.log
│   │   └── experiment_n32.log
│   ├── Kitchenham/
│   ├── Radjenovic/
│   └── Wahono/
├── SMOTE_treatment_name_1/
│   ├── Hall/
│   │   ├── results_nb_sk_al_smote_n8.csv
│   │   ├── results_nb_sk_al_smote_n16.csv
│   │   ├── results_nb_sk_al_smote_n32.csv
│   │   ├── experiment_n8.log
│   │   ├── experiment_n16.log
│   │   └── experiment_n32.log
│   ├── Kitchenham/
│   ├── Radjenovic/
│   └── Wahono/
├── treatment_name_2/
│   └── ...
└── ...
```

## Key Features

- **Automatic n_pos values**: Each experiment automatically runs for n_pos = 8, 16, and 32
- **Dynamic treatment discovery**: The Makefile automatically discovers all treatment directories from `../data/`
- **Simplified commands**: No need to specify individual n_pos values or complex target names
- **Background execution**: All experiments run in the background using `nohup`
- **Virtual environment**: Automatically creates and uses a Python virtual environment
- **Step cutoff control**: Experiments can be limited to a specific number of steps using `step_cutoffs.json` configuration file

## Available Treatments

The following preprocessing treatments are available:
- tfidf_100_max_features_10000_top_n_100
- tfidf_50_max_features_10000_top_n_50
- tfidf_chi2_100_max_features_10000_top_n_100
- tfidf_chi2_50_max_features_10000_top_n_50
- tfidf_chi2_svc_100_max_features_10000_random_state_42_svc_C_1.0_top_n_100
- tfidf_chi2_svc_50_max_features_10000_random_state_42_svc_C_1.0_top_n_50
- tfidf_chi2_svc_max_features_5000_random_state_42_svc_C_1.0_top_n_25
- tfidf_ig_100_max_features_10000_random_state_42_top_n_100
- tfidf_ig_50_max_features_10000_random_state_42_top_n_50
- tfidf_autoencoder_100_batch_size_32_epochs_50_learning_rate_0.001_max_features_10000_random_state_42_top_n_100
- tfidf_autoencoder_50_batch_size_32_epochs_50_learning_rate_0.001_max_features_10000_random_state_42_top_n_50
- tfidf_autoencoder_50_fixed_batch_size_32_epochs_50_learning_rate_0.001_max_features_10000_random_state_42_top_n_50
- tfidf_autoencoder_batch_size_32_epochs_50_learning_rate_0.001_max_features_5000_random_state_42_top_n_30
- tfidf_autoencoder_test_batch_size_64_epochs_20_learning_rate_0.001_max_features_5000_random_state_42_top_n_25

## Available Datasets

- Hall
- Kitchenham
- Radjenovic
- Wahono

## Experiment Parameters

All experiments use the following parameters:
- n_pos: 8 (number of positive samples to start with)
- repeats: 20 (number of repeats)
- batch_size: 1000 (batch size for acquisition)
- step_cutoff: Dataset-specific (controlled by `step_cutoffs.json` in the main directory)

## Step Cutoff Configuration

The experiments now support running until a specific step count instead of acquiring all available samples. This is controlled by the `step_cutoffs.json` file in the main project directory:

```json
{
  "Hall": 50,
  "Kitchenham": 30,
  "Radjenovic": 40,
  "Wahono": 60,
  "default": 100
}
```

For more details, see `README_step_cutoff.md`.

## Notes

- All experiments run in the background using `nohup`
- Log files are created for each experiment in the results directory
- Use `make status` to check which experiments are currently running
- Use `make clean` to remove log files if needed
