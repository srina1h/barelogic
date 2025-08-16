#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def run_experiment(data_file, output_file, n_pos=8, repeats=20, batch_size=1000):
    """Run naive_bayes_AL.py on a single dataset"""
    cmd = [
        sys.executable, 
        'new_models/naive_bayes_AL.py',
        '--input', data_file,
        '--output', output_file,
        '--n_pos', str(n_pos),
        '--repeats', str(repeats),
        '--batch_size', str(batch_size)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully completed: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment for {data_file}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Define the datasets
    datasets = ['Hall.csv', 'Kitchenham.csv', 'Radjenovic.csv', 'Wahono.csv']
    
    # Get all preprocessing treatment folders
    data_dir = Path('data')
    treatment_folders = [f for f in data_dir.iterdir() if f.is_dir() and f.name != 'raw']
    
    # Create results directory structure
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Process each treatment
    for treatment_folder in treatment_folders:
        treatment_name = treatment_folder.name
        print(f"\n{'='*60}")
        print(f"Processing treatment: {treatment_name}")
        print(f"{'='*60}")
        
        # Create treatment subfolder in results
        treatment_results_dir = results_dir / treatment_name
        treatment_results_dir.mkdir(exist_ok=True)
        
        # Process each dataset
        for dataset in datasets:
            data_file = treatment_folder / dataset
            if not data_file.exists():
                print(f"Warning: {data_file} not found, skipping...")
                continue
                
            print(f"\nProcessing dataset: {dataset}")
            
            # Create dataset subfolder
            dataset_name = dataset.replace('.csv', '')
            dataset_results_dir = treatment_results_dir / dataset_name
            dataset_results_dir.mkdir(exist_ok=True)
            
            # Run experiment
            output_file = dataset_results_dir / 'results_nb_sk_al.csv'
            
            success = run_experiment(
                str(data_file), 
                str(output_file),
                n_pos=8,
                repeats=20,
                batch_size=1000
            )
            
            if success:
                print(f"✓ Completed: {output_file}")
            else:
                print(f"✗ Failed: {dataset} in {treatment_name}")
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved in: {results_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
