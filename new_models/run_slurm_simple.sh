#!/bin/bash
#SBATCH --job-name=barelogic_experiments
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/ssrini27/se/barelogic/new_models

# Create virtual environment first (only once)
echo "Setting up virtual environment..."
make venv

# Run regular experiments and wait for them to complete
echo "Starting regular experiments..."
make all
make wait-regular

# Run SMOTE experiments and wait for them to complete
echo "Starting SMOTE experiments..."
make all-smote
make wait-smote

echo "All experiments completed!"

# Check results
echo "Checking results..."
find ../results -name "*.csv" | wc -l
echo "CSV files found:"
find ../results -name "*.csv" | head -10

echo "Job completed successfully!"
