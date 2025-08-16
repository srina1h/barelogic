import sys
import os

# Add the current directory to the path to import the base preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tfidf_base import TFIDFBasePreprocessor, create_argument_parser
from feature_selectors import Chi2SVCCoefficientSelector
from feature_utils import save_chi2_svc_features, save_feature_analysis

class TFIDFChi2SVCPreprocessor(TFIDFBasePreprocessor):
    """
    TF-IDF with Chi2+SVC coefficient-based feature selection that extracts top N features
    using a two-stage approach: Chi-squared selection followed by Linear SVC coefficient ranking.
    """
    
    def __init__(self, 
                 top_n_features: int = 1000,
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2,
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 1),
                 chi2_k: int = None,
                 random_state: int = 42,
                 svc_C: float = 1.0):
        """
        Initialize TF-IDF with Chi2+SVC preprocessor.
        
        Args:
            top_n_features: Number of final features to select
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
            max_features: Maximum features for TF-IDF vectorizer
            ngram_range: N-gram range for TF-IDF vectorizer
            chi2_k: Number of features to select with Chi-squared (if None, uses top_n_features*2)
            random_state: Random state for reproducibility
            svc_C: Regularization parameter for Linear SVC
        """
        super().__init__(top_n_features, remove_stopwords, use_stemming, 
                        use_lemmatization, min_word_length, max_features, ngram_range)
        self.chi2_k = chi2_k
        self.random_state = random_state
        self.svc_C = svc_C
    
    def select_features(self, tfidf_matrix, labels):
        """
        Select top N features using Chi-squared followed by Linear SVC coefficients.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Target labels
            
        Returns:
            Selected feature matrix
        """
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Use Chi2+SVC selector
        selector = Chi2SVCCoefficientSelector(
            k=self.top_n_features,
            chi2_k=self.chi2_k,
            random_state=self.random_state,
            C=self.svc_C
        )
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
        
        # Add Chi2+SVC specific parameters
        dir_hyperparams['random_state'] = self.random_state
        dir_hyperparams['svc_C'] = self.svc_C
        if self.chi2_k is not None:
            dir_hyperparams['chi2_k'] = self.chi2_k
        
        all_hyperparams['random_state'] = self.random_state
        all_hyperparams['svc_C'] = self.svc_C
        if self.chi2_k is not None:
            all_hyperparams['chi2_k'] = self.chi2_k
        
        return dir_hyperparams, all_hyperparams
    
    def process_datasets(self, data_dir: str = "data/raw", output_dir: str = "data/tfidf_chi2_svc") -> dict:
        """
        Process all datasets using TF-IDF with Chi2+SVC feature selection.
        
        Args:
            data_dir: Directory containing raw CSV files
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary of processed datasets
        """
        # Use parent class method
        processed_datasets, actual_output_dir = super().process_datasets(
            data_dir, output_dir, "_tfidf_chi2_svc"
        )
        
        # Save feature information
        save_chi2_svc_features(self.selected_features, self.feature_scores, actual_output_dir)
        save_feature_analysis(self.selected_features, self.feature_scores, actual_output_dir)
        
        return processed_datasets

def main():
    """Main function for command line usage."""
    parser = create_argument_parser(
        'TF-IDF with Chi2+SVC coefficient-based feature selection for text datasets',
        'data/tfidf_chi2_svc'
    )
    
    # Add Chi2+SVC specific arguments
    parser.add_argument('--chi2_k', type=int, default=None,
                       help='Number of features to select with Chi-squared (default: top_n*2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--svc_C', type=float, default=1.0,
                       help='Regularization parameter for Linear SVC (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TFIDFChi2SVCPreprocessor(
        top_n_features=args.top_n,
        max_features=args.max_features,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        use_lemmatization=args.use_lemmatization,
        min_word_length=args.min_word_length,
        chi2_k=args.chi2_k,
        random_state=args.random_state,
        svc_C=args.svc_C
    )
    
    # Process datasets
    processed_datasets = preprocessor.process_datasets(args.input_dir, args.output_dir)
    
    print(f"TF-IDF with Chi2+SVC feature selection preprocessing completed. Processed {len(processed_datasets)} datasets.")

if __name__ == "__main__":
    main()
