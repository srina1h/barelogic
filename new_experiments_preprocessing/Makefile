# Makefile for text preprocessing experiments
# Usage examples:
#   make tfidf TOP_N=500 MAX_FEATURES=5000
#   make tfidf_ig TOP_N=1000 MAX_FEATURES=10000
#   make clean
#   make all

# Default parameters
TOP_N ?= 1000
MAX_FEATURES ?= 10000
MIN_WORD_LENGTH ?= 2
RANDOM_STATE ?= 42
INPUT_DIR ?= ../data/raw
TFIDF_OUTPUT_DIR ?= ../data/tfidf
TFIDF_IG_OUTPUT_DIR ?= ../data/tfidf_ig
TFIDF_CHI2_OUTPUT_DIR ?= ../data/tfidf_chi2
TFIDF_CHI2_SVC_OUTPUT_DIR ?= ../data/tfidf_chi2_svc
TFIDF_AUTOENCODER_OUTPUT_DIR ?= ../data/tfidf_autoencoder

# Python interpreter
PYTHON := venv/bin/python

# Preprocessing scripts
TFIDF_SCRIPT := tfidf.py
TFIDF_IG_SCRIPT := tfidf_ig.py
TFIDF_CHI2_SCRIPT := tfidf_chi2.py
TFIDF_CHI2_SVC_SCRIPT := tfidf_chi2_svc.py
TFIDF_AUTOENCODER_SCRIPT := tfidf_autoencoder.py

# Default target
.PHONY: all
all: tfidf tfidf_ig tfidf_chi2 tfidf_chi2_svc tfidf_autoencoder

# TF-IDF preprocessing
.PHONY: tfidf
tfidf:
	@echo "Running TF-IDF preprocessing with TOP_N=$(TOP_N), MAX_FEATURES=$(MAX_FEATURES)"
	$(PYTHON) $(TFIDF_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_OUTPUT_DIR)

# TF-IDF with Information Gain preprocessing
.PHONY: tfidf_ig
tfidf_ig:
	@echo "Running TF-IDF with Information Gain preprocessing with TOP_N=$(TOP_N), MAX_FEATURES=$(MAX_FEATURES)"
	$(PYTHON) $(TFIDF_IG_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_IG_OUTPUT_DIR)

# TF-IDF with Chi-squared preprocessing
.PHONY: tfidf_chi2
tfidf_chi2:
	@echo "Running TF-IDF with Chi-squared preprocessing with TOP_N=$(TOP_N), MAX_FEATURES=$(MAX_FEATURES)"
	$(PYTHON) $(TFIDF_CHI2_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_OUTPUT_DIR)

# TF-IDF with Chi2+SVC preprocessing
.PHONY: tfidf_chi2_svc
tfidf_chi2_svc:
	@echo "Running TF-IDF with Chi2+SVC preprocessing with TOP_N=$(TOP_N), MAX_FEATURES=$(MAX_FEATURES)"
	$(PYTHON) $(TFIDF_CHI2_SVC_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--svc_C 1.0 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_SVC_OUTPUT_DIR)

# TF-IDF with Autoencoder preprocessing
.PHONY: tfidf_autoencoder
tfidf_autoencoder:
	@echo "Running TF-IDF with Autoencoder preprocessing with TOP_N=$(TOP_N), MAX_FEATURES=$(MAX_FEATURES)"
	$(PYTHON) $(TFIDF_AUTOENCODER_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--epochs 50 \
		--batch_size 32 \
		--learning_rate 0.001 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_AUTOENCODER_OUTPUT_DIR)

# Quick preprocessing with smaller feature sets (for testing)
.PHONY: quick
quick: quick_tfidf quick_tfidf_ig quick_tfidf_chi2 quick_tfidf_chi2_svc quick_tfidf_autoencoder

.PHONY: quick_tfidf
quick_tfidf:
	@echo "Running quick TF-IDF preprocessing (TOP_N=100, MAX_FEATURES=1000)"
	$(PYTHON) $(TFIDF_SCRIPT) \
		--top_n 100 \
		--max_features 1000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_OUTPUT_DIR)_quick

.PHONY: quick_tfidf_ig
quick_tfidf_ig:
	@echo "Running quick TF-IDF with Information Gain preprocessing (TOP_N=100, MAX_FEATURES=1000)"
	$(PYTHON) $(TFIDF_IG_SCRIPT) \
		--top_n 100 \
		--max_features 1000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_IG_OUTPUT_DIR)_quick

.PHONY: quick_tfidf_chi2
quick_tfidf_chi2:
	@echo "Running quick TF-IDF with Chi-squared preprocessing (TOP_N=100, MAX_FEATURES=1000)"
	$(PYTHON) $(TFIDF_CHI2_SCRIPT) \
		--top_n 100 \
		--max_features 1000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_OUTPUT_DIR)_quick

.PHONY: quick_tfidf_chi2_svc
quick_tfidf_chi2_svc:
	@echo "Running quick TF-IDF with Chi2+SVC preprocessing (TOP_N=100, MAX_FEATURES=1000)"
	$(PYTHON) $(TFIDF_CHI2_SVC_SCRIPT) \
		--top_n 100 \
		--max_features 1000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--svc_C 1.0 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_SVC_OUTPUT_DIR)_quick

.PHONY: quick_tfidf_autoencoder
quick_tfidf_autoencoder:
	@echo "Running quick TF-IDF with Autoencoder preprocessing (TOP_N=100, MAX_FEATURES=1000)"
	$(PYTHON) $(TFIDF_AUTOENCODER_SCRIPT) \
		--top_n 100 \
		--max_features 1000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--epochs 30 \
		--batch_size 64 \
		--learning_rate 0.001 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_AUTOENCODER_OUTPUT_DIR)_quick

# Large feature set preprocessing
.PHONY: large
large: large_tfidf large_tfidf_ig large_tfidf_chi2 large_tfidf_chi2_svc large_tfidf_autoencoder

.PHONY: large_tfidf
large_tfidf:
	@echo "Running large TF-IDF preprocessing (TOP_N=5000, MAX_FEATURES=50000)"
	$(PYTHON) $(TFIDF_SCRIPT) \
		--top_n 5000 \
		--max_features 50000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_OUTPUT_DIR)_large

.PHONY: large_tfidf_ig
large_tfidf_ig:
	@echo "Running large TF-IDF with Information Gain preprocessing (TOP_N=5000, MAX_FEATURES=50000)"
	$(PYTHON) $(TFIDF_IG_SCRIPT) \
		--top_n 5000 \
		--max_features 50000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_IG_OUTPUT_DIR)_large

.PHONY: large_tfidf_chi2
large_tfidf_chi2:
	@echo "Running large TF-IDF with Chi-squared preprocessing (TOP_N=5000, MAX_FEATURES=50000)"
	$(PYTHON) $(TFIDF_CHI2_SCRIPT) \
		--top_n 5000 \
		--max_features 50000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_OUTPUT_DIR)_large

.PHONY: large_tfidf_chi2_svc
large_tfidf_chi2_svc:
	@echo "Running large TF-IDF with Chi2+SVC preprocessing (TOP_N=5000, MAX_FEATURES=50000)"
	$(PYTHON) $(TFIDF_CHI2_SVC_SCRIPT) \
		--top_n 5000 \
		--max_features 50000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--svc_C 1.0 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_CHI2_SVC_OUTPUT_DIR)_large

.PHONY: large_tfidf_autoencoder
large_tfidf_autoencoder:
	@echo "Running large TF-IDF with Autoencoder preprocessing (TOP_N=5000, MAX_FEATURES=50000)"
	$(PYTHON) $(TFIDF_AUTOENCODER_SCRIPT) \
		--top_n 5000 \
		--max_features 50000 \
		--min_word_length $(MIN_WORD_LENGTH) \
		--random_state $(RANDOM_STATE) \
		--epochs 100 \
		--batch_size 128 \
		--learning_rate 0.001 \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_AUTOENCODER_OUTPUT_DIR)_large

# Preprocessing with different text cleaning options
.PHONY: no_stemming
no_stemming:
	@echo "Running TF-IDF preprocessing without stemming"
	$(PYTHON) $(TFIDF_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_OUTPUT_DIR)_no_stemming \
		--no-use_stemming

.PHONY: with_lemmatization
with_lemmatization:
	@echo "Running TF-IDF preprocessing with lemmatization"
	$(PYTHON) $(TFIDF_SCRIPT) \
		--top_n $(TOP_N) \
		--max_features $(MAX_FEATURES) \
		--min_word_length $(MIN_WORD_LENGTH) \
		--input_dir $(INPUT_DIR) \
		--output_dir $(TFIDF_OUTPUT_DIR)_lemmatized \
		--use_lemmatization

# Check if input data exists
.PHONY: check_data
check_data:
	@echo "Checking input data..."
	@if [ ! -d "$(INPUT_DIR)" ]; then \
		echo "Error: Input directory $(INPUT_DIR) does not exist"; \
		exit 1; \
	fi
	@echo "Found input directory: $(INPUT_DIR)"
	@echo "Available datasets:"
	@ls -la $(INPUT_DIR)/*.csv 2>/dev/null || echo "No CSV files found in $(INPUT_DIR)"

# Show preprocessing statistics
.PHONY: stats
stats:
	@echo "Preprocessing Statistics:"
	@echo "========================"
	@echo "TF-IDF output directories:"
	@for dir in ../data/tfidf*; do \
		if [ -d "$$dir" ]; then \
			csv_count=$$(ls "$$dir"/*.csv 2>/dev/null | wc -l); \
			echo "  - $$(basename "$$dir"): $$csv_count CSV files"; \
		fi; \
	done
	@echo "TF-IDF-IG output directories:"
	@for dir in ../data/tfidf_ig*; do \
		if [ -d "$$dir" ]; then \
			csv_count=$$(ls "$$dir"/*.csv 2>/dev/null | wc -l); \
			echo "  - $$(basename "$$dir"): $$csv_count CSV files"; \
		fi; \
	done
	@echo "TF-IDF-Chi2 output directories:"
	@for dir in ../data/tfidf_chi2*; do \
		if [ -d "$$dir" ]; then \
			csv_count=$$(ls "$$dir"/*.csv 2>/dev/null | wc -l); \
			echo "  - $$(basename "$$dir"): $$csv_count CSV files"; \
		fi; \
	done
	@echo "TF-IDF-Chi2-SVC output directories:"
	@for dir in ../data/tfidf_chi2_svc*; do \
		if [ -d "$$dir" ]; then \
			csv_count=$$(ls "$$dir"/*.csv 2>/dev/null | wc -l); \
			echo "  - $$(basename "$$dir"): $$csv_count CSV files"; \
		fi; \
	done
	@echo "TF-IDF-Autoencoder output directories:"
	@for dir in ../data/tfidf_autoencoder*; do \
		if [ -d "$$dir" ]; then \
			csv_count=$$(ls "$$dir"/*.csv 2>/dev/null | wc -l); \
			echo "  - $$(basename "$$dir"): $$csv_count CSV files"; \
		fi; \
	done

# Clean up generated files
.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(TFIDF_OUTPUT_DIR)
	@rm -rf $(TFIDF_IG_OUTPUT_DIR)
	@echo "Cleanup completed."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all              - Run all preprocessing methods (default)"
	@echo "  tfidf            - Run TF-IDF preprocessing"
	@echo "  tfidf_ig         - Run TF-IDF with Information Gain preprocessing"
	@echo "  tfidf_chi2       - Run TF-IDF with Chi-squared preprocessing"
	@echo "  tfidf_chi2_svc   - Run TF-IDF with Chi2+SVC preprocessing"
	@echo "  tfidf_autoencoder - Run TF-IDF with Autoencoder preprocessing"
	@echo "  quick            - Run quick preprocessing with smaller feature sets"
	@echo "  large            - Run preprocessing with large feature sets"
	@echo "  no_stemming      - Run TF-IDF without stemming"
	@echo "  with_lemmatization - Run TF-IDF with lemmatization"
	@echo "  check_data       - Check if input data exists"
	@echo "  stats            - Show preprocessing statistics"
	@echo "  clean            - Clean up generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Parameters (can be overridden):"
	@echo "  TOP_N=$(TOP_N)              - Number of top features to select"
	@echo "  MAX_FEATURES=$(MAX_FEATURES)    - Maximum features for TF-IDF vectorizer"
	@echo "  MIN_WORD_LENGTH=$(MIN_WORD_LENGTH)  - Minimum word length"
	@echo "  RANDOM_STATE=$(RANDOM_STATE)    - Random state for reproducibility"
	@echo "  INPUT_DIR=$(INPUT_DIR)        - Input directory with raw CSV files"
	@echo "  TFIDF_OUTPUT_DIR=$(TFIDF_OUTPUT_DIR)    - TF-IDF output directory"
	@echo "  TFIDF_IG_OUTPUT_DIR=$(TFIDF_IG_OUTPUT_DIR) - TF-IDF-IG output directory"
	@echo "  TFIDF_CHI2_OUTPUT_DIR=$(TFIDF_CHI2_OUTPUT_DIR) - TF-IDF-Chi2 output directory"
	@echo "  TFIDF_CHI2_SVC_OUTPUT_DIR=$(TFIDF_CHI2_SVC_OUTPUT_DIR) - TF-IDF-Chi2-SVC output directory"
	@echo "  TFIDF_AUTOENCODER_OUTPUT_DIR=$(TFIDF_AUTOENCODER_OUTPUT_DIR) - TF-IDF-Autoencoder output directory"
	@echo ""
	@echo "Usage examples:"
	@echo "  make tfidf TOP_N=500 MAX_FEATURES=5000"
	@echo "  make tfidf_ig TOP_N=1000 MAX_FEATURES=10000"
	@echo "  make tfidf_chi2 TOP_N=1000 MAX_FEATURES=10000"
	@echo "  make tfidf_chi2_svc TOP_N=1000 MAX_FEATURES=10000"
	@echo "  make tfidf_autoencoder TOP_N=1000 MAX_FEATURES=10000"
	@echo "  make quick"
	@echo "  make clean"
