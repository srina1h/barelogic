import pandas as pd
import numpy as np
import re
import os
from typing import List, Tuple, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DataPreprocessor:
    """
    Base class for data preprocessing with text cleaning, tokenization, 
    stemming, lemmatization, and stop word removal.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stop words
            use_stemming: Whether to apply Porter stemming
            use_lemmatization: Whether to apply lemmatization
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Store processed data
        self.processed_texts = []
        self.labels = []
        self.dataset_names = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, extra whitespace, etc.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize text and apply cleaning operations.
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Apply cleaning operations
        cleaned_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
                
            # Skip if it's a stop word
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming if enabled
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            # Apply lemmatization if enabled
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def load_dataset(self, file_path: str, dataset_name: str = None) -> pd.DataFrame:
        """
        Load a dataset and extract abstract and label columns.
        
        Args:
            file_path: Path to the CSV file
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with processed abstract and label columns
        """
        print(f"Loading dataset: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Determine the correct column names based on the dataset structure
        abstract_col = None
        label_col = None
        
        # Check for abstract column (case insensitive)
        for col in df.columns:
            if 'abstract' in col.lower():
                abstract_col = col
                break
        
        # Check for label column (case insensitive)
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if abstract_col is None:
            raise ValueError(f"No abstract column found in {file_path}")
        if label_col is None:
            raise ValueError(f"No label column found in {file_path}")
        
        # Extract and process the data
        abstracts = df[abstract_col].fillna('')
        labels = df[label_col].fillna('no')
        
        # Convert labels to binary (0/1)
        labels_binary = (labels.str.lower() == 'yes').astype(int)
        
        # Process abstracts
        processed_abstracts = []
        for abstract in abstracts:
            tokens = self.tokenize_and_clean(abstract)
            processed_abstracts.append(' '.join(tokens))
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'abstract': processed_abstracts,
            'label': labels_binary,
            'original_abstract': abstracts,
            'original_label': labels
        })
        
        # Store for later use
        self.processed_texts.extend(processed_abstracts)
        self.labels.extend(labels_binary.tolist())
        
        if dataset_name:
            self.dataset_names.extend([dataset_name] * len(result_df))
        
        print(f"Loaded {len(result_df)} samples from {file_path}")
        print(f"Label distribution: {result_df['label'].value_counts().to_dict()}")
        
        return result_df
    
    def load_all_datasets(self, data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """
        Load all datasets from the raw data directory.
        
        Args:
            data_dir: Directory containing raw CSV files
            
        Returns:
            Dictionary mapping dataset names to processed DataFrames
        """
        datasets = {}
        
        for file_path in os.listdir(data_dir):
            if file_path.endswith('.csv'):
                full_path = os.path.join(data_dir, file_path)
                dataset_name = file_path.replace('.csv', '')
                
                try:
                    df = self.load_dataset(full_path, dataset_name)
                    datasets[dataset_name] = df
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return datasets
    
    def save_processed_data(self, datasets: Dict[str, pd.DataFrame], 
                          output_dir: str, 
                          filename_suffix: str = "",
                          dir_hyperparams: Dict[str, Any] = None,
                          all_hyperparams: Dict[str, Any] = None):
        """
        Save processed datasets to the specified directory.
        
        Args:
            datasets: Dictionary of processed DataFrames
            output_dir: Output directory
            filename_suffix: Suffix to add to filenames
            dir_hyperparams: Dictionary of hyperparameters for directory naming
            all_hyperparams: Dictionary of all hyperparameters for the text file
        """
        # Append hyperparameters to directory name if provided
        if dir_hyperparams:
            param_str = "_".join([f"{k}_{v}" for k, v in sorted(dir_hyperparams.items())])
            output_dir = f"{output_dir}_{param_str}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, df in datasets.items():
            # Use original dataset name without suffix
            filename = f"{dataset_name}.csv"
            output_path = os.path.join(output_dir, filename)
            
            # Save all columns including features
            df.to_csv(output_path, index=False)
            
            print(f"Saved {len(df)} samples to {output_path}")
        
        # Save hyperparameter information
        if all_hyperparams:
            param_file = os.path.join(output_dir, "hyperparameters.txt")
            with open(param_file, 'w') as f:
                f.write("Hyperparameters used for preprocessing:\n")
                f.write("=" * 50 + "\n")
                for key, value in sorted(all_hyperparams.items()):
                    f.write(f"{key}: {value}\n")
            print(f"Hyperparameters saved to {param_file}")
        
        return output_dir
    
    def get_vocabulary(self) -> List[str]:
        """
        Get the vocabulary from all processed texts.
        
        Returns:
            List of unique words
        """
        all_words = set()
        for text in self.processed_texts:
            words = text.split()
            all_words.update(words)
        
        return sorted(list(all_words))
    
    def get_text_features(self) -> Tuple[List[str], List[int]]:
        """
        Get processed texts and labels for feature extraction.
        
        Returns:
            Tuple of (processed_texts, labels)
        """
        return self.processed_texts, self.labels
