# Text Preprocessing Pipeline for Naive Bayes Experiments

This directory contains a comprehensive text preprocessing pipeline designed for naive Bayes experiments with the four datasets (Hall, Kitchenham, Radjenovic, Wahono).

## Overview

The preprocessing pipeline consists of:

1. **Base Preprocessing** (`load_and_prep.py`): Text cleaning, tokenization, stemming, lemmatization, and stop word removal
2. **TF-IDF Preprocessing** (`tfidf.py`): Extracts top N TF-IDF features
3. **TF-IDF with Information Gain** (`tfidf_ig.py`): Extracts top N features based on information gain scores
4. **TF-IDF with Chi-squared** (`tfidf_chi2.py`): Extracts top N features based on chi-squared statistical test

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The pipeline will automatically download required NLTK data on first run.

## Data Structure

The pipeline expects raw CSV files in `data/raw/` with the following structure:
- **Abstract column**: Contains the text to be processed
- **Label column**: Contains binary labels (yes/no converted to 1/0)

The pipeline automatically detects the correct column names for different datasets.

## Usage

### Using the Makefile (Recommended)

The Makefile provides convenient targets for different preprocessing scenarios:

```bash
# Run all preprocessing methods with default parameters
make all

# Run only TF-IDF preprocessing
make tfidf

# Run only TF-IDF with Information Gain preprocessing
make tfidf_ig

# Run only TF-IDF with Chi-squared preprocessing
make tfidf_chi2

# Run quick preprocessing (smaller feature sets for testing)
make quick

# Run large feature set preprocessing
make large

# Custom parameters
make tfidf TOP_N=500 MAX_FEATURES=5000
make tfidf_ig TOP_N=1000 MAX_FEATURES=10000
make tfidf_chi2 TOP_N=1000 MAX_FEATURES=10000

# Check available data
make check_data

# Show preprocessing statistics
make stats

# Clean up generated files
make clean

# Show help
make help
```

### Using Python Scripts Directly

You can also run the preprocessing scripts directly:

```bash
# TF-IDF preprocessing
python tfidf.py --top_n 1000 --max_features 10000 --input_dir data/raw --output_dir data/tfidf

# TF-IDF with Information Gain preprocessing
python tfidf_ig.py --top_n 1000 --max_features 10000 --input_dir data/raw --output_dir data/tfidf_ig

# TF-IDF with Chi-squared preprocessing
python tfidf_chi2.py --top_n 1000 --max_features 10000 --input_dir data/raw --output_dir data/tfidf_chi2
```

## Preprocessing Techniques

### 1. Base Preprocessing (`DataPreprocessor`)

**Features:**
- Text cleaning (HTML removal, special character removal)
- Tokenization using NLTK
- Porter stemming
- WordNet lemmatization (optional)
- Stop word removal
- Minimum word length filtering

**Parameters:**
- `remove_stopwords`: Whether to remove stop words (default: True)
- `use_stemming`: Whether to apply Porter stemming (default: True)
- `use_lemmatization`: Whether to apply lemmatization (default: False)
- `min_word_length`: Minimum word length to keep (default: 2)

### 2. TF-IDF Preprocessing (`TFIDFPreprocessor`)

**Features:**
- Extends base preprocessing
- TF-IDF vectorization with configurable parameters
- Top N feature selection based on TF-IDF scores
- Support for unigrams and bigrams

**Parameters:**
- `top_n_features`: Number of top features to select (default: 1000)
- `max_features`: Maximum features for TF-IDF vectorizer (default: 10000)
- `min_df`: Minimum document frequency (default: 2)
- `max_df`: Maximum document frequency (default: 0.95)
- `ngram_range`: N-gram range (default: (1, 2))
- `sublinear_tf`: Apply sublinear TF scaling (default: True)

### 3. TF-IDF with Information Gain (`TFIDFInformationGainPreprocessor`)

**Features:**
- Extends base preprocessing
- TF-IDF vectorization
- Feature selection based on mutual information (information gain)
- Detailed feature analysis and ranking

**Parameters:**
- `top_n_features`: Number of top features to select (default: 1000)
- `max_features`: Maximum features for TF-IDF vectorizer (default: 10000)
- `random_state`: Random state for reproducibility (default: 42)

## Output Structure

Each preprocessing method creates output directories with the following structure:

```
data/
├── tfidf/
│   ├── Hall.csv
│   ├── Kitchenham.csv
│   ├── Radjenovic.csv
│   ├── Wahono.csv
│   └── tfidf_features.txt
└── tfidf_ig/
    ├── Hall.csv
    ├── Kitchenham.csv
    ├── Radjenovic.csv
    ├── Wahono.csv
    ├── tfidf_ig_features.txt
    └── feature_analysis.csv
```

### Output File Format

Each processed CSV file contains:
- **Feature columns**: Named as `tfidf_<index>_<feature_name>` or `tfidf_ig_<index>_<feature_name>`
- **label**: Binary target variable (0/1)
- **abstract**: Original processed abstract text

**Note**: Files are saved with their original names (e.g., `Hall.csv`, `Kitchenham.csv`) and are differentiated by the directory they are stored in (e.g., `data/tfidf/Hall.csv` vs `data/tfidf_ig/Hall.csv`).

### Feature Information Files

- `tfidf_features.txt`: List of top TF-IDF features
- `tfidf_ig_features.txt`: List of top features with information gain scores
- `tfidf_chi2_features.txt`: List of top features with chi-squared scores
- `feature_analysis.csv`: Detailed feature analysis with scores and rankings

## Configuration

### Default Parameters

The following parameters can be customized:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_N` | 1000 | Number of top features to select |
| `MAX_FEATURES` | 10000 | Maximum features for TF-IDF vectorizer |
| `MIN_WORD_LENGTH` | 2 | Minimum word length |
| `RANDOM_STATE` | 42 | Random state for reproducibility |
| `INPUT_DIR` | data/raw | Input directory with raw CSV files |
| `TFIDF_OUTPUT_DIR` | data/tfidf | TF-IDF output directory |
| `TFIDF_IG_OUTPUT_DIR` | data/tfidf_ig | TF-IDF-IG output directory |
| `TFIDF_CHI2_OUTPUT_DIR` | data/tfidf_chi2 | TF-IDF-Chi2 output directory |

### Text Cleaning Options

The pipeline supports various text cleaning configurations:

```bash
# Without stemming
make no_stemming

# With lemmatization
make with_lemmatization

# Custom parameters
make tfidf --no-use_stemming --use_lemmatization
```

## Examples

### Quick Testing
```bash
# Run quick preprocessing for testing
make quick
```

### Production Run
```bash
# Run with large feature sets
make large
```

### Custom Configuration
```bash
# Custom feature selection
make tfidf TOP_N=500 MAX_FEATURES=5000
make tfidf_ig TOP_N=2000 MAX_FEATURES=20000
make tfidf_chi2 TOP_N=2000 MAX_FEATURES=20000

# Different text cleaning
make tfidf --no-use_stemming --use_lemmatization
```

## Troubleshooting

### Common Issues

1. **NLTK data not found**: The pipeline will automatically download required NLTK data on first run.

2. **Memory issues with large datasets**: Use smaller `MAX_FEATURES` values or run the `quick` target first.

3. **Input data not found**: Use `make check_data` to verify your input directory structure.

### Performance Tips

- For large datasets, start with the `quick` target to test the pipeline
- Use `MAX_FEATURES` to limit memory usage
- Consider using `--no-use_stemming` for faster processing if stemming is not critical

## Extending the Pipeline

To add new preprocessing techniques:

1. Create a new Python file (e.g., `new_method.py`)
2. Extend the `DataPreprocessor` class
3. Implement the required methods
4. Add corresponding targets to the Makefile

Example:
```python
from load_and_prep import DataPreprocessor

class NewMethodPreprocessor(DataPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your initialization code
    
    def process_datasets(self, data_dir, output_dir):
        # Implement your preprocessing logic
        pass
```

## License

This preprocessing pipeline is part of the barelogic project.
