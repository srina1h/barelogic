#!/usr/bin/env python3
"""
Test script to verify processed data can be loaded correctly.
"""

import pandas as pd
import os
import sys

def test_processed_data():
    """Test loading and basic analysis of processed data."""
    
    # Test directories
    test_dirs = [
        "../data/tfidf",
        "../data/tfidf_ig",
        "../data/tfidf_chi2",
        "../data/tfidf_chi2_svc",
        "../data/tfidf_autoencoder"
    ]
    
    for test_dir in test_dirs:
        # Check for directories with hyperparameter suffixes
        matching_dirs = []
        if os.path.exists(test_dir):
            matching_dirs.append(test_dir)
        else:
            # Look for directories that start with the base name
            base_name = os.path.basename(test_dir)
            parent_dir = os.path.dirname(test_dir)
            if os.path.exists(parent_dir):
                for item in os.listdir(parent_dir):
                    if item.startswith(base_name) and os.path.isdir(os.path.join(parent_dir, item)):
                        matching_dirs.append(os.path.join(parent_dir, item))
        
        if not matching_dirs:
            print(f"No directories found for {test_dir}, skipping...")
            continue
        
        for actual_dir in matching_dirs:
            print(f"\n{'='*60}")
            print(f"Testing directory: {actual_dir}")
            print(f"{'='*60}")
            
            # List CSV files directly in the directory
            csv_files = [f for f in os.listdir(actual_dir) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files:")
        
        for csv_file in csv_files:
            file_path = os.path.join(actual_dir, csv_file)
            print(f"\n  File: {csv_file}")
            
            try:
                # Load the data
                df = pd.read_csv(file_path)
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {len(df.columns)}")
                
                # Check for required columns
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    print(f"    Label distribution: {label_counts.to_dict()}")
                
                if 'abstract' in df.columns:
                    print(f"    Abstract column present: Yes")
                
                # Count feature columns
                feature_cols = [col for col in df.columns if col.startswith('tfidf')]
                print(f"    Feature columns: {len(feature_cols)}")
                
                if feature_cols:
                    # Show some feature statistics
                    feature_data = df[feature_cols]
                    print(f"    Feature matrix shape: {feature_data.shape}")
                    print(f"    Feature sparsity: {(feature_data == 0).sum().sum() / feature_data.size:.2%}")
                    print(f"    Feature value range: [{feature_data.min().min():.4f}, {feature_data.max().max():.4f}]")
                
            except Exception as e:
                print(f"    Error loading {csv_file}: {e}")
        
        # Check for feature info files
        feature_files = [f for f in os.listdir(actual_dir) if f.endswith('.txt') or f.endswith('.csv') and 'feature' in f]
        if feature_files:
            print(f"\n  Feature information files: {feature_files}")
            
            # Show top features if available
            feature_txt = [f for f in feature_files if f.endswith('.txt') and 'feature' in f]
            if feature_txt:
                feature_file = os.path.join(actual_dir, feature_txt[0])
                try:
                    with open(feature_file, 'r') as f:
                        lines = f.readlines()
                        print(f"    Top 10 features from {feature_txt[0]}:")
                        for i, line in enumerate(lines[3:13]):  # Skip header
                            if line.strip():
                                print(f"      {line.strip()}")
                except Exception as e:
                    print(f"    Error reading feature file: {e}")

def main():
    """Main function."""
    print("Testing Processed Data")
    print("=" * 60)
    
    test_processed_data()
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
