import os
import pandas as pd
from typing import List, Dict, Any

def save_feature_info(features: List[str], scores: List[float], 
                     output_dir: str, filename: str, 
                     title: str, score_name: str = "Score") -> str:
    """
    Save feature information to a text file.
    
    Args:
        features: List of feature names
        scores: List of feature scores
        output_dir: Output directory
        filename: Output filename
        title: Title for the feature list
        score_name: Name of the score column
        
    Returns:
        Path to the saved file
    """
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<30} {score_name:<12}\n")
        f.write("-" * 60 + "\n")
        
        for i, (feature, score) in enumerate(zip(features, scores)):
            f.write(f"{i+1:<6} {feature:<30} {score:<12.6f}\n")
    
    print(f"Feature information saved to {file_path}")
    return file_path

def save_feature_analysis(features: List[str], scores: List[float], 
                         output_dir: str, filename: str = "feature_analysis.csv") -> str:
    """
    Save detailed feature analysis to a CSV file.
    
    Args:
        features: List of feature names
        scores: List of feature scores
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to the saved file
    """
    file_path = os.path.join(output_dir, filename)
    
    analysis_df = pd.DataFrame({
        'feature': features,
        'score': scores,
        'rank': range(1, len(features) + 1)
    })
    analysis_df.to_csv(file_path, index=False)
    
    print(f"Feature analysis saved to {file_path}")
    return file_path

def save_tfidf_features(features: List[str], output_dir: str) -> str:
    """
    Save TF-IDF feature information.
    
    Args:
        features: List of feature names
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    return save_feature_info(
        features=features,
        scores=range(len(features)),  # Dummy scores for TF-IDF
        output_dir=output_dir,
        filename="tfidf_features.txt",
        title=f"Top {len(features)} TF-IDF features:",
        score_name="TF-IDF"
    )

def save_information_gain_features(features: List[str], scores: List[float], 
                                 output_dir: str) -> str:
    """
    Save information gain feature information.
    
    Args:
        features: List of feature names
        scores: List of information gain scores
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    return save_feature_info(
        features=features,
        scores=scores,
        output_dir=output_dir,
        filename="tfidf_ig_features.txt",
        title=f"Top {len(features)} TF-IDF features by Information Gain:",
        score_name="Info_Gain"
    )

def save_chi2_features(features: List[str], scores: List[float], 
                       output_dir: str) -> str:
    """
    Save chi-squared feature information.
    
    Args:
        features: List of feature names
        scores: List of chi-squared scores
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    return save_feature_info(
        features=features,
        scores=scores,
        output_dir=output_dir,
        filename="tfidf_chi2_features.txt",
        title=f"Top {len(features)} TF-IDF features by Chi-squared test:",
        score_name="Chi2_Score"
    )

def save_chi2_svc_features(features: List[str], scores: List[float], 
                          output_dir: str) -> str:
    """
    Save Chi2+SVC feature information.
    
    Args:
        features: List of feature names
        scores: List of SVC coefficient scores
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    return save_feature_info(
        features=features,
        scores=scores,
        output_dir=output_dir,
        filename="tfidf_chi2_svc_features.txt",
        title=f"Top {len(features)} TF-IDF features by Chi2+SVC coefficients:",
        score_name="SVC_Coeff"
    )

def save_autoencoder_features(features: List[str], scores: List[float], 
                            output_dir: str) -> str:
    """
    Save autoencoder feature information.
    
    Args:
        features: List of feature names
        scores: List of autoencoder reconstruction error scores
        output_dir: Output directory
        
    Returns:
        Path to the saved file
    """
    return save_feature_info(
        features=features,
        scores=scores,
        output_dir=output_dir,
        filename="tfidf_autoencoder_features.txt",
        title=f"Top {len(features)} TF-IDF features by Autoencoder reconstruction error:",
        score_name="Recon_Error"
    )
