import sys
import os

# Add the current directory to the path to import the base preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tfidf_base import TFIDFBasePreprocessor, create_argument_parser
from feature_selectors import AutoencoderFeatureSelector
from feature_utils import save_autoencoder_features, save_feature_analysis

class TFIDFAutoencoderPreprocessor(TFIDFBasePreprocessor):
    """
    TF-IDF with Autoencoder-based feature selection that extracts top N features
    using neural network autoencoders to learn feature importance.
    """
    
    def __init__(self, 
                 top_n_features: int = 1000,
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2,
                 max_features: int = 10000,
                 ngram_range: tuple = (1, 1),
                 hidden_dim: int = None,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize TF-IDF with Autoencoder preprocessor.
        
        Args:
            top_n_features: Number of features to select
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
            max_features: Maximum features for TF-IDF vectorizer
            ngram_range: N-gram range for TF-IDF vectorizer
            hidden_dim: Hidden dimension for autoencoder (if None, uses top_n_features*2)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            random_state: Random state for reproducibility
        """
        super().__init__(top_n_features, remove_stopwords, use_stemming, 
                        use_lemmatization, min_word_length, max_features, ngram_range)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
    
    def select_features(self, tfidf_matrix, labels):
        """
        Select top N features using autoencoder-based feature importance.
        
        Args:
            tfidf_matrix: TF-IDF feature matrix
            labels: Target labels (not used for autoencoder)
            
        Returns:
            Selected feature matrix
        """
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Use autoencoder selector
        selector = AutoencoderFeatureSelector(
            k=self.top_n_features,
            hidden_dim=self.hidden_dim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_state=self.random_state
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
        
        # Add autoencoder specific parameters
        dir_hyperparams['random_state'] = self.random_state
        dir_hyperparams['epochs'] = self.epochs
        dir_hyperparams['batch_size'] = self.batch_size
        dir_hyperparams['learning_rate'] = self.learning_rate
        if self.hidden_dim is not None:
            dir_hyperparams['hidden_dim'] = self.hidden_dim
        
        all_hyperparams['random_state'] = self.random_state
        all_hyperparams['epochs'] = self.epochs
        all_hyperparams['batch_size'] = self.batch_size
        all_hyperparams['learning_rate'] = self.learning_rate
        if self.hidden_dim is not None:
            all_hyperparams['hidden_dim'] = self.hidden_dim
        
        return dir_hyperparams, all_hyperparams
    
    def process_datasets(self, data_dir: str = "data/raw", output_dir: str = "data/tfidf_autoencoder") -> dict:
        """
        Process all datasets using TF-IDF with Autoencoder feature selection.
        
        Args:
            data_dir: Directory containing raw CSV files
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary of processed datasets
        """
        # Use parent class method
        processed_datasets, actual_output_dir = super().process_datasets(
            data_dir, output_dir, "_tfidf_autoencoder"
        )
        
        # Save feature information
        save_autoencoder_features(self.selected_features, self.feature_scores, actual_output_dir)
        save_feature_analysis(self.selected_features, self.feature_scores, actual_output_dir)
        
        return processed_datasets

def main():
    """Main function for command line usage."""
    parser = create_argument_parser(
        'TF-IDF with Autoencoder-based feature selection for text datasets',
        'data/tfidf_autoencoder'
    )
    
    # Add autoencoder specific arguments
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension for autoencoder (default: top_n*2)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = TFIDFAutoencoderPreprocessor(
        top_n_features=args.top_n,
        max_features=args.max_features,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        use_lemmatization=args.use_lemmatization,
        min_word_length=args.min_word_length,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.random_state
    )
    
    # Process datasets
    processed_datasets = preprocessor.process_datasets(args.input_dir, args.output_dir)
    
    print(f"TF-IDF with Autoencoder feature selection preprocessing completed. Processed {len(processed_datasets)} datasets.")

if __name__ == "__main__":
    main()
