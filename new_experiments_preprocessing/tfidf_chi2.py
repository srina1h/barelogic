import sys
import os

# Add the current directory to the path to import the base preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tfidf_base import TFIDFBasePreprocessor, create_argument_parser
from feature_selectors import ChiSquaredSelector
from feature_utils import save_chi2_features, save_feature_analysis

class TFIDFChi2Preprocessor(TFIDFBasePreprocessor):
    """
    TF-IDF with Chi-squared feature selection that extracts top N features
    based on chi-squared statistical test.
    """
    
    def select_features(self, tfidf_matrix, labels):
        """
        Select top N features based on chi-squared statistical test.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Target labels
            
        Returns:
            Selected feature matrix
        """
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Use chi-squared selector
        selector = ChiSquaredSelector(self.top_n_features)
        selected_matrix, self.selected_features, self.feature_scores = selector.select(
            tfidf_matrix, labels, feature_names
        )
        
        return selected_matrix
    
    def process_datasets(self, data_dir: str = "data/raw", output_dir: str = "data/tfidf_chi2") -> dict:
        """
        Process all datasets using TF-IDF with Chi-squared feature selection.
        
        Args:
            data_dir: Directory containing raw CSV files
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary of processed datasets
        """
        # Use parent class method
        processed_datasets, actual_output_dir = super().process_datasets(
            data_dir, output_dir, "_tfidf_chi2"
        )
        
        # Save feature information
        save_chi2_features(self.selected_features, self.feature_scores, actual_output_dir)
        save_feature_analysis(self.selected_features, self.feature_scores, actual_output_dir)
        
        return processed_datasets

def main():
    """Main function for command line usage."""
    parser = create_argument_parser(
        'TF-IDF with Chi-squared feature selection for text datasets',
        'data/tfidf_chi2'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TFIDFChi2Preprocessor(
        top_n_features=args.top_n,
        max_features=args.max_features,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        use_lemmatization=args.use_lemmatization,
        min_word_length=args.min_word_length
    )
    
    # Process datasets
    processed_datasets = preprocessor.process_datasets(args.input_dir, args.output_dir)
    
    print(f"TF-IDF with Chi-squared feature selection preprocessing completed. Processed {len(processed_datasets)} datasets.")

if __name__ == "__main__":
    main()
