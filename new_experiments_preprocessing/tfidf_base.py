import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import os
import sys

# Add the current directory to the path to import the base preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_and_prep import DataPreprocessor

class TFIDFBasePreprocessor(DataPreprocessor):
    """
    Base class for TF-IDF preprocessing methods that provides common functionality
    for TF-IDF feature extraction and selection.
    """
    
    def __init__(self, 
                 top_n_features: int = 1000,
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2,
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 1)):
        """
        Initialize base TF-IDF preprocessor.
        
        Args:
            top_n_features: Number of top features to select
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
            max_features: Maximum features for TF-IDF vectorizer
            ngram_range: N-gram range for TF-IDF vectorizer
        """
        super().__init__(remove_stopwords, use_stemming, use_lemmatization, min_word_length)
        
        self.top_n_features = top_n_features
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.selected_features = []
        self.feature_scores = []
        
    def fit_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer.
        
        Args:
            texts: List of processed text documents
            
        Returns:
            TF-IDF feature matrix
        """
        print(f"Fitting TF-IDF vectorizer with max_features={self.max_features}")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            ngram_range=self.ngram_range,  # Configurable n-gram range
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def select_features(self, tfidf_matrix: np.ndarray, labels: List[int]) -> np.ndarray:
        """
        Abstract method for feature selection. Must be implemented by subclasses.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Target labels
            
        Returns:
            Selected feature matrix
        """
        raise NotImplementedError("Subclasses must implement select_features method")
    
    def get_feature_names(self, prefix: str = "tfidf") -> List[str]:
        """
        Get feature names with prefix.
        
        Args:
            prefix: Prefix for feature names
            
        Returns:
            List of feature names
        """
        return [f"{prefix}_{i}_{feature}" for i, feature in enumerate(self.selected_features)]
    
    def prepare_hyperparameters(self, method_suffix: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare hyperparameters for directory naming and text file.
        
        Args:
            method_suffix: Suffix to add to method name
            
        Returns:
            Tuple of (directory_hyperparams, all_hyperparams)
        """
        # Directory hyperparameters (only variable parameters)
        dir_hyperparams = {
            'top_n': self.top_n_features,
            'max_features': self.max_features
        }
        
        # All hyperparameters for the text file
        all_hyperparams = {
            'top_n': self.top_n_features,
            'max_features': self.max_features,
            'remove_stopwords': self.remove_stopwords,
            'use_stemming': self.use_stemming,
            'use_lemmatization': self.use_lemmatization,
            'min_word_length': self.min_word_length,
            'ngram_range': self.ngram_range
        }
        
        return dir_hyperparams, all_hyperparams
    
    def process_datasets(self, data_dir: str = "data/raw", output_dir: str = "data/tfidf", 
                        method_suffix: str = "_tfidf") -> Dict[str, pd.DataFrame]:
        """
        Process all datasets using TF-IDF with feature selection.
        
        Args:
            data_dir: Directory containing raw CSV files
            output_dir: Output directory for processed data
            method_suffix: Suffix for method name
            
        Returns:
            Dictionary of processed datasets
        """
        print(f"Starting TF-IDF preprocessing with {method_suffix}...")
        
        # Load all datasets
        datasets = self.load_all_datasets(data_dir)
        
        if not datasets:
            print("No datasets loaded!")
            return {}
        
        # Get all texts and labels
        all_texts, all_labels = self.get_text_features()
        
        if not all_texts:
            print("No texts to process!")
            return {}
        
        # Fit TF-IDF
        tfidf_matrix = self.fit_tfidf(all_texts)
        
        # Select features (implemented by subclasses)
        selected_matrix = self.select_features(tfidf_matrix, all_labels)
        
        # Convert to DataFrame and split by dataset
        feature_names = self.get_feature_names(method_suffix.replace("_", ""))
        all_features_df = pd.DataFrame(selected_matrix, columns=feature_names)
        
        # Add labels
        all_features_df['label'] = all_labels
        
        # Split back into individual datasets
        processed_datasets = {}
        start_idx = 0
        
        for dataset_name, original_df in datasets.items():
            n_samples = len(original_df)
            end_idx = start_idx + n_samples
            
            # Extract features for this dataset
            dataset_features = all_features_df.iloc[start_idx:end_idx].copy()
            
            # Add original abstract for reference
            dataset_features['abstract'] = original_df['abstract'].values
            
            processed_datasets[dataset_name] = dataset_features
            start_idx = end_idx
        
        # Prepare hyperparameters
        dir_hyperparams, all_hyperparams = self.prepare_hyperparameters()
        
        # Save processed data and get the actual output directory
        actual_output_dir = self.save_processed_data(processed_datasets, output_dir, method_suffix, dir_hyperparams, all_hyperparams)
        
        return processed_datasets, actual_output_dir

def create_argument_parser(description: str, default_output_dir: str) -> argparse.ArgumentParser:
    """
    Create a standardized argument parser for TF-IDF preprocessing scripts.
    
    Args:
        description: Description for the argument parser
        default_output_dir: Default output directory
        
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--top_n', type=int, default=1000, 
                       help='Number of top features to select (default: 1000)')
    parser.add_argument('--max_features', type=int, default=10000,
                       help='Maximum features for TF-IDF vectorizer (default: 10000)')
    parser.add_argument('--remove_stopwords', action='store_true', default=True,
                       help='Remove stop words (default: True)')
    parser.add_argument('--use_stemming', action='store_true', default=True,
                       help='Apply Porter stemming (default: True)')
    parser.add_argument('--use_lemmatization', action='store_true', default=False,
                       help='Apply lemmatization (default: False)')
    parser.add_argument('--min_word_length', type=int, default=2,
                       help='Minimum word length (default: 2)')
    parser.add_argument('--input_dir', type=str, default='data/raw',
                       help='Input directory with raw CSV files (default: data/raw)')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help=f'Output directory for processed data (default: {default_output_dir})')
    
    return parser
