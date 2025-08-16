import sys
import os

# Add the current directory to the path to import the base preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tfidf_base import TFIDFBasePreprocessor, create_argument_parser
from feature_selectors import InformationGainSelector
from feature_utils import save_information_gain_features, save_feature_analysis

class TFIDFInformationGainPreprocessor(TFIDFBasePreprocessor):
    """
    TF-IDF with Information Gain preprocessing that extracts top N features
    based on mutual information classification.
    """
    
    def __init__(self, 
                 top_n_features: int = 1000,
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2,
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 1),
                 random_state: int = 42):
        """
        Initialize TF-IDF with Information Gain preprocessor.
        
        Args:
            top_n_features: Number of top features to select based on information gain
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
            max_features: Maximum features for TF-IDF vectorizer
            ngram_range: N-gram range for TF-IDF vectorizer
            random_state: Random state for reproducibility
        """
        super().__init__(top_n_features, remove_stopwords, use_stemming, 
                        use_lemmatization, min_word_length, max_features, ngram_range)
        self.random_state = random_state
    
    def select_features(self, tfidf_matrix, labels):
        """
        Select top N features based on information gain.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Target labels
            
        Returns:
            Selected feature matrix
        """
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Use information gain selector
        selector = InformationGainSelector(self.top_n_features, self.random_state)
        selected_matrix, self.selected_features, self.feature_scores = selector.select(
            tfidf_matrix, labels, feature_names
        )
        
        return selected_matrix
    
    def prepare_hyperparameters(self, method_suffix: str = "") -> tuple:
        """
        Prepare hyperparameters for directory naming and text file.
        
        Args:
            method_suffix: Suffix to add to method name
            
        Returns:
            Tuple of (directory_hyperparams, all_hyperparams)
        """
        # Get base hyperparameters
        dir_hyperparams, all_hyperparams = super().prepare_hyperparameters(method_suffix)
        
        # Add random_state to both
        dir_hyperparams['random_state'] = self.random_state
        all_hyperparams['random_state'] = self.random_state
        
        return dir_hyperparams, all_hyperparams
    
    def process_datasets(self, data_dir: str = "data/raw", output_dir: str = "data/tfidf_ig") -> dict:
        """
        Process all datasets using TF-IDF with Information Gain preprocessing.
        
        Args:
            data_dir: Directory containing raw CSV files
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary of processed datasets
        """
        # Use parent class method
        processed_datasets, actual_output_dir = super().process_datasets(
            data_dir, output_dir, "_tfidf_ig"
        )
        
        # Save feature information
        save_information_gain_features(self.selected_features, self.feature_scores, actual_output_dir)
        save_feature_analysis(self.selected_features, self.feature_scores, actual_output_dir)
        
        return processed_datasets

def main():
    """Main function for command line usage."""
    parser = create_argument_parser(
        'TF-IDF with Information Gain preprocessing for text datasets',
        'data/tfidf_ig'
    )
    
    # Add random_state argument
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TFIDFInformationGainPreprocessor(
        top_n_features=args.top_n,
        max_features=args.max_features,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        use_lemmatization=args.use_lemmatization,
        min_word_length=args.min_word_length,
        random_state=args.random_state
    )
    
    # Process datasets
    processed_datasets = preprocessor.process_datasets(args.input_dir, args.output_dir)
    
    print(f"TF-IDF with Information Gain preprocessing completed. Processed {len(processed_datasets)} datasets.")

if __name__ == "__main__":
    main()
