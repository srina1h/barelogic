import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FeatureSelector:
    """
    Base class for feature selection methods.
    """
    
    def __init__(self, k: int):
        """
        Initialize feature selector.
        
        Args:
            k: Number of features to select
        """
        self.k = k
        self.selector = None
        self.selected_features = []
        self.feature_scores = []
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select features.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        raise NotImplementedError("Subclasses must implement select method")

class TopKSelector(FeatureSelector):
    """
    Select top K features based on TF-IDF scores.
    """
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select top K features based on TF-IDF scores.
        
        Args:
            X: TF-IDF feature matrix
            y: Target labels (not used for this selector)
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        # Calculate TF-IDF scores (mean across documents)
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Use mean TF-IDF scores as feature importance
        feature_scores = np.mean(X_dense, axis=0)
        
        # Get top K features
        top_indices = np.argsort(feature_scores)[::-1][:self.k]
        
        # Select features
        selected_matrix = X_dense[:, top_indices]
        self.selected_features = [feature_names[i] for i in top_indices]
        self.feature_scores = feature_scores[top_indices]
        
        print(f"Selected {len(self.selected_features)} features based on TF-IDF scores")
        print(f"Selected feature matrix shape: {selected_matrix.shape}")
        print(f"TF-IDF scores range: {min(self.feature_scores):.4f} - {max(self.feature_scores):.4f}")
        
        return selected_matrix, self.selected_features, self.feature_scores

class InformationGainSelector(FeatureSelector):
    """
    Select top K features based on information gain (mutual information).
    """
    
    def __init__(self, k: int, random_state: int = 42):
        """
        Initialize information gain selector.
        
        Args:
            k: Number of features to select
            random_state: Random state for reproducibility
        """
        super().__init__(k)
        self.random_state = random_state
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select top K features based on information gain.
        
        Args:
            X: TF-IDF feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        print(f"Calculating information gain for {X.shape[1]} features")
        
        # Convert to dense matrix if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_dense, y, random_state=self.random_state)
        
        # Get top K features
        top_indices = np.argsort(mi_scores)[::-1][:self.k]
        
        # Select features
        selected_matrix = X_dense[:, top_indices]
        self.selected_features = [feature_names[i] for i in top_indices]
        self.feature_scores = mi_scores[top_indices]
        
        print(f"Selected {len(self.selected_features)} features based on information gain")
        print(f"Selected feature matrix shape: {selected_matrix.shape}")
        print(f"Information gain scores range: {min(self.feature_scores):.4f} - {max(self.feature_scores):.4f}")
        
        return selected_matrix, self.selected_features, self.feature_scores

class ChiSquaredSelector(FeatureSelector):
    """
    Select top K features based on chi-squared statistical test.
    """
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select top K features based on chi-squared test.
        
        Args:
            X: TF-IDF feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        print(f"Selecting top {self.k} features using chi-squared test")
        
        # Convert to dense matrix if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Ensure non-negative values for chi2 test
        X_dense = np.abs(X_dense)
        
        # Calculate chi-squared scores
        chi2_scores = chi2(X_dense, y)[0]
        
        # Get top K features
        top_indices = np.argsort(chi2_scores)[::-1][:self.k]
        
        # Select features
        selected_matrix = X_dense[:, top_indices]
        self.selected_features = [feature_names[i] for i in top_indices]
        self.feature_scores = chi2_scores[top_indices]
        
        print(f"Selected {len(self.selected_features)} features based on chi-squared test")
        print(f"Selected feature matrix shape: {selected_matrix.shape}")
        print(f"Chi-squared scores range: {min(self.feature_scores):.4f} - {max(self.feature_scores):.4f}")
        
        return selected_matrix, self.selected_features, self.feature_scores

class Chi2SVCCoefficientSelector(FeatureSelector):
    """
    Two-stage feature selection: Chi-squared followed by Linear SVC coefficient ranking.
    """
    
    def __init__(self, k: int, chi2_k: int = None, random_state: int = 42, C: float = 1.0):
        """
        Initialize Chi2+SVC coefficient selector.
        
        Args:
            k: Number of final features to select
            chi2_k: Number of features to select with Chi-squared (if None, uses k*2)
            random_state: Random state for reproducibility
            C: Regularization parameter for Linear SVC
        """
        super().__init__(k)
        self.chi2_k = chi2_k if chi2_k is not None else k * 2
        self.random_state = random_state
        self.C = C
        self.chi2_selector = None
        self.svc_model = None
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select features using Chi-squared followed by Linear SVC coefficients.
        
        Args:
            X: TF-IDF feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        print(f"Stage 1: Selecting top {self.chi2_k} features using chi-squared test")
        
        # Convert to dense matrix if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Ensure non-negative values for chi2 test
        X_dense = np.abs(X_dense)
        
        # Stage 1: Chi-squared selection
        chi2_scores = chi2(X_dense, y)[0]
        chi2_top_indices = np.argsort(chi2_scores)[::-1][:self.chi2_k]
        
        # Get Chi-squared selected features
        X_chi2_selected = X_dense[:, chi2_top_indices]
        chi2_feature_names = [feature_names[i] for i in chi2_top_indices]
        chi2_feature_scores = chi2_scores[chi2_top_indices]
        
        print(f"Stage 1 completed: Selected {len(chi2_feature_names)} features with Chi-squared")
        print(f"Chi-squared scores range: {min(chi2_feature_scores):.4f} - {max(chi2_feature_scores):.4f}")
        
        # Stage 2: Linear SVC coefficient ranking
        print(f"Stage 2: Training Linear SVC and ranking features by coefficients")
        
        # Standardize features for SVC
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_chi2_selected)
        
        # Train Linear SVC
        self.svc_model = LinearSVC(C=self.C, random_state=self.random_state, max_iter=1000)
        self.svc_model.fit(X_scaled, y)
        
        # Get feature importance from coefficients
        coefficients = np.abs(self.svc_model.coef_[0])
        
        # Get top K features based on SVC coefficients
        svc_top_indices = np.argsort(coefficients)[::-1][:self.k]
        
        # Select final features
        selected_matrix = X_chi2_selected[:, svc_top_indices]
        self.selected_features = [chi2_feature_names[i] for i in svc_top_indices]
        self.feature_scores = coefficients[svc_top_indices]
        
        print(f"Stage 2 completed: Selected {len(self.selected_features)} features based on SVC coefficients")
        print(f"Selected feature matrix shape: {selected_matrix.shape}")
        print(f"SVC coefficient scores range: {min(self.feature_scores):.4f} - {max(self.feature_scores):.4f}")
        
        return selected_matrix, self.selected_features, self.feature_scores

class AutoencoderFeatureSelector(FeatureSelector):
    """
    Feature selection using autoencoders to learn feature importance.
    """
    
    def __init__(self, k: int, hidden_dim: int = None, epochs: int = 50, 
                 batch_size: int = 32, learning_rate: float = 0.001, 
                 random_state: int = 42):
        """
        Initialize autoencoder feature selector.
        
        Args:
            k: Number of features to select
            hidden_dim: Hidden dimension for autoencoder (if None, uses k*2)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            random_state: Random state for reproducibility
        """
        super().__init__(k)
        self.hidden_dim = hidden_dim if hidden_dim is not None else k * 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.autoencoder = None
        self.scaler = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_autoencoder(self, input_dim: int) -> nn.Module:
        """
        Create autoencoder model.
        
        Args:
            input_dim: Input dimension
            
        Returns:
            Autoencoder model
        """
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, input_dim),
                    nn.ReLU()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return Autoencoder(input_dim, self.hidden_dim)
    
    def _train_autoencoder(self, X_tensor: torch.Tensor) -> nn.Module:
        """
        Train autoencoder model.
        
        Args:
            X_tensor: Input tensor
            
        Returns:
            Trained autoencoder model
        """
        # Create model
        autoencoder = self._create_autoencoder(X_tensor.shape[1])
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder reconstructs input
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        
        # Set random seed for training
        torch.manual_seed(self.random_state)
        
        # Training loop
        autoencoder.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = autoencoder(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        return autoencoder
    
    def _compute_feature_importance(self, autoencoder: nn.Module, X_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute feature importance using reconstruction error.
        
        Args:
            autoencoder: Trained autoencoder model
            X_tensor: Input tensor
            
        Returns:
            Feature importance scores
        """
        autoencoder.eval()
        with torch.no_grad():
            # Get reconstruction
            reconstructed = autoencoder(X_tensor)
            
            # Compute reconstruction error for each feature
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=0)
            
            # Convert to numpy and normalize
            feature_importance = reconstruction_errors.numpy()
            
        return feature_importance
    
    def select(self, X: np.ndarray, y: List[int], feature_names: List[str]) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Select features using autoencoder-based feature importance.
        
        Args:
            X: TF-IDF feature matrix
            y: Target labels (not used for autoencoder)
            feature_names: List of feature names
            
        Returns:
            Tuple of (selected_matrix, selected_feature_names, feature_scores)
        """
        print(f"Training autoencoder for feature selection (hidden_dim={self.hidden_dim}, epochs={self.epochs})")
        
        # Convert to dense matrix if needed
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_dense)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Train autoencoder
        self.autoencoder = self._train_autoencoder(X_tensor)
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance(self.autoencoder, X_tensor)
        
        # Get top K features
        top_indices = np.argsort(feature_importance)[::-1][:self.k]
        
        # Select features
        selected_matrix = X_dense[:, top_indices]
        self.selected_features = [feature_names[i] for i in top_indices]
        self.feature_scores = feature_importance[top_indices]
        
        print(f"Selected {len(self.selected_features)} features based on autoencoder reconstruction error")
        print(f"Selected feature matrix shape: {selected_matrix.shape}")
        print(f"Feature importance scores range: {min(self.feature_scores):.6f} - {max(self.feature_scores):.6f}")
        
        return selected_matrix, self.selected_features, self.feature_scores
