"""
DeepFM model for no-show prediction
Combines wide (linear) and deep (neural network) components for tabular data
Based on DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Optional, Any, Tuple, List
import joblib
import logging
from pathlib import Path
import copy

from config.config import RANDOM_SEED, EARLY_STOPPING_ROUNDS

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DeepFMNet(nn.Module):
    """
    DeepFM Network combining linear and deep components
    """
    
    def __init__(self, field_dims: List[int], embedding_dim: int, 
                 hidden_dims: List[int], dropout_rate: float = 0.3):
        """
        Initialize DeepFM network
        
        Args:
            field_dims: List of field dimensions for embeddings
            embedding_dim: Embedding dimension
            hidden_dims: List of hidden layer dimensions for deep part
            dropout_rate: Dropout probability
        """
        super(DeepFMNet, self).__init__()
        
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.num_fields = len(field_dims)
        
        # Linear part (FM first-order)
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(field_dim, 1) for field_dim in field_dims
        ])
        
        # FM second-order interactions
        self.fm_embeddings = nn.ModuleList([
            nn.Embedding(field_dim, embedding_dim) for field_dim in field_dims
        ])
        
        # Deep part
        self.deep_input_dim = self.num_fields * embedding_dim
        
        deep_layers = []
        prev_dim = self.deep_input_dim
        
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Initialize embeddings
        for embedding in self.linear_embeddings:
            nn.init.normal_(embedding.weight, mean=0, std=0.01)
        for embedding in self.fm_embeddings:
            nn.init.normal_(embedding.weight, mean=0, std=0.01)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_fields)
        """
        # Linear part (first-order)
        linear_outputs = []
        for i, embedding in enumerate(self.linear_embeddings):
            linear_outputs.append(embedding(x[:, i]))
        
        linear_output = torch.sum(torch.cat(linear_outputs, dim=1), dim=1, keepdim=True)
        
        # FM part (second-order)
        fm_outputs = []
        for i, embedding in enumerate(self.fm_embeddings):
            fm_outputs.append(embedding(x[:, i]))
        
        fm_matrix = torch.stack(fm_outputs, dim=1)  # (batch_size, num_fields, embedding_dim)
        
        # FM second-order interaction: 0.5 * ((sum)^2 - sum(square))
        square_of_sum = torch.pow(torch.sum(fm_matrix, dim=1), 2)  # (batch_size, embedding_dim)
        sum_of_square = torch.sum(torch.pow(fm_matrix, 2), dim=1)  # (batch_size, embedding_dim)
        
        fm_output = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Deep part
        deep_input = fm_matrix.view(-1, self.deep_input_dim)  # Flatten embeddings
        deep_output = self.deep_network(deep_input)
        
        # Combine all parts
        output = linear_output + fm_output + deep_output
        
        return output


class DeepFMNoShow:
    """
    DeepFM classifier for no-show prediction
    Combines wide (linear) and deep (neural network) components
    """
    
    def __init__(self, **params):
        """
        Initialize DeepFM model
        
        Args:
            **params: Model hyperparameters
        """
        # Default parameters
        self.params = {
            'embedding_dim': 16,
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 512,
            'epochs': 100,
            'weight_decay': 1e-4,
            **params
        }
        
        self.model = None
        self.feature_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.field_dims = None
        self.num_features = None
        
    def _encode_features(self, X: np.ndarray, fit_encoders: bool = False) -> torch.Tensor:
        """
        Encode features for DeepFM (convert to categorical indices)
        
        Args:
            X: Features array
            fit_encoders: Whether to fit the encoders
            
        Returns:
            Encoded features tensor
        """
        X_encoded = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            feature_name = f'feature_{i}'
            
            if fit_encoders:
                # Create encoder for this feature
                encoder = LabelEncoder()
                
                # For continuous features, bin them into categories
                if len(np.unique(X[:, i])) > 20:  # Assume continuous if >20 unique values
                    # Bin continuous features into 20 bins
                    bins = np.percentile(X[:, i], np.linspace(0, 100, 21))
                    bins = np.unique(bins)  # Remove duplicates
                    X_binned = np.digitize(X[:, i], bins) - 1
                    X_binned = np.clip(X_binned, 0, len(bins) - 2)
                else:
                    # Keep categorical features as is
                    X_binned = X[:, i].astype(int)
                
                X_encoded[:, i] = encoder.fit_transform(X_binned)
                self.feature_encoders[feature_name] = encoder
                
            else:
                encoder = self.feature_encoders[feature_name]
                
                # Apply same binning as during fit
                if len(encoder.classes_) > 20:  # Was continuous
                    bins = np.percentile(X[:, i], np.linspace(0, 100, 21))
                    bins = np.unique(bins)
                    X_binned = np.digitize(X[:, i], bins) - 1
                    X_binned = np.clip(X_binned, 0, len(bins) - 2)
                else:
                    X_binned = X[:, i].astype(int)
                
                # Handle unseen categories
                try:
                    X_encoded[:, i] = encoder.transform(X_binned)
                except ValueError:
                    # Handle unseen categories by assigning to most frequent class
                    X_encoded_feature = np.zeros(len(X_binned))
                    for j, val in enumerate(X_binned):
                        if val in encoder.classes_:
                            X_encoded_feature[j] = encoder.transform([val])[0]
                        else:
                            X_encoded_feature[j] = 0  # Default to first class
                    X_encoded[:, i] = X_encoded_feature.astype(int)
        
        return torch.LongTensor(X_encoded).to(self.device)
    
    def _calculate_pos_weight(self, y: np.ndarray) -> torch.Tensor:
        """
        Calculate positive class weight for imbalanced data
        
        Args:
            y: Target array
            
        Returns:
            Positive weight tensor
        """
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        pos_weight = neg_count / pos_count
        
        logger.info(f"Class distribution: {neg_count} negatives, {pos_count} positives")
        logger.info(f"Positive weight: {pos_weight:.4f}")
        
        return torch.FloatTensor([pos_weight]).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the model with optional validation for early stopping
        
        Args:
            X: Training features
            y: Training target
            feature_names: Optional feature names
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
        """
        logger.info(f"Training DeepFM with params: {self.params}")
        logger.info(f"Device: {self.device}")
        
        self.feature_names = feature_names
        self.num_features = X.shape[1]
        
        # Encode features
        X_encoded = self._encode_features(X, fit_encoders=True)
        
        # Calculate field dimensions for embeddings
        self.field_dims = []
        for i in range(self.num_features):
            encoder = self.feature_encoders[f'feature_{i}']
            self.field_dims.append(len(encoder.classes_))
        
        logger.info(f"Field dimensions: {self.field_dims}")
        
        # Initialize model
        self.model = DeepFMNet(
            field_dims=self.field_dims,
            embedding_dim=self.params['embedding_dim'],
            hidden_dims=self.params['hidden_dims'],
            dropout_rate=self.params['dropout_rate']
        ).to(self.device)
        
        # Prepare target
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_encoded, y_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True
        )
        
        # Prepare validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_encoded = self._encode_features(X_val, fit_encoders=False)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_encoded, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)
        
        # Loss function with class weights
        pos_weight = self._calculate_pos_weight(y)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.params['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
                
                # Early stopping
                if patience_counter >= EARLY_STOPPING_ROUNDS:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        logger.info("DeepFM training completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_encoded = self._encode_features(X, fit_encoders=False)
        
        with torch.no_grad():
            outputs = self.model(X_encoded)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Probability of positive class (no-show)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_encoded = self._encode_features(X, fit_encoders=False)
        
        with torch.no_grad():
            outputs = self.model(X_encoded)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        return probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (approximated using embedding norms)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # For DeepFM, approximate importance using linear embedding weights
        importance_scores = []
        
        for i, linear_embedding in enumerate(self.model.linear_embeddings):
            # Use L2 norm of linear embeddings as importance proxy
            weight_norm = torch.norm(linear_embedding.weight).item()
            importance_scores.append(weight_norm)
        
        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        importance_scores = importance_scores / np.sum(importance_scores)
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance_scores))],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'params': self.params,
            'feature_encoders': self.feature_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'field_dims': self.field_dims,
            'num_features': self.num_features
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        save_dict = joblib.load(filepath)
        
        self.params = save_dict['params']
        self.feature_encoders = save_dict['feature_encoders']
        self.feature_names = save_dict['feature_names']
        self.is_fitted = save_dict['is_fitted']
        self.field_dims = save_dict['field_dims']
        self.num_features = save_dict['num_features']
        
        if save_dict['model_state_dict'] and self.field_dims:
            self.model = DeepFMNet(
                field_dims=self.field_dims,
                embedding_dim=self.params['embedding_dim'],
                hidden_dims=self.params['hidden_dims'],
                dropout_rate=self.params['dropout_rate']
            ).to(self.device)
            self.model.load_state_dict(save_dict['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    @staticmethod
    def suggest_params(trial) -> Dict:
        """
        Suggest parameters for Optuna trial (optimized for efficiency)
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        # Optimize key hyperparameters for DeepFM
        params = {}
        
        # Embedding dimension
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [8, 16, 32])
        
        # Network architecture - limit to 3 variations for efficiency
        arch_choice = trial.suggest_categorical('architecture', ['small', 'medium', 'large'])
        if arch_choice == 'small':
            params['hidden_dims'] = [128, 64]
        elif arch_choice == 'medium':
            params['hidden_dims'] = [256, 128, 64]
        else:  # large
            params['hidden_dims'] = [512, 256, 128]
        
        # Key hyperparameters
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.5)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [256, 512, 1024])
        
        return params
    
    @staticmethod
    def is_available() -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False