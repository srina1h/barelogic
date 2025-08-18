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

# Run regular experiments first
echo "Starting regular experiments..."
make all

# Wait for regular experiments to complete
echo "Waiting for regular experiments to complete..."
sleep 30  # Give some time for processes to start

# Check if regular experiments are still running
while pgrep -f "naive_bayes_AL.py" > /dev/null; do
    echo "Regular experiments still running... waiting"
    sleep 60
done

echo "Regular experiments completed!"

# Run SMOTE experiments
echo "Starting SMOTE experiments..."
make all-smote

# Wait for SMOTE experiments to complete
echo "Waiting for SMOTE experiments to complete..."
sleep 30  # Give some time for processes to start

# Check if SMOTE experiments are still running
while pgrep -f "naive_bayes_AL_SMOTE.py" > /dev/null; do
    echo "SMOTE experiments still running... waiting"
    sleep 60
done

echo "All experiments completed!"

# Check results
echo "Checking results..."
find ../results -name "*.csv" | wc -l
echo "CSV files found:"
find ../results -name "*.csv" | head -10

echo "Job completed successfully!"
