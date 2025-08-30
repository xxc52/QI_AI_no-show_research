"""
Multi-Layer Perceptron (MLP) model for no-show prediction
PyTorch-based neural network optimized for tabular data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, Tuple
import joblib
import logging
from pathlib import Path
import copy

from config.config import RANDOM_SEED, EARLY_STOPPING_ROUNDS

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MLPNet(nn.Module):
    """
    Multi-Layer Perceptron Network for tabular data
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float = 0.3):
        """
        Initialize MLP network
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(MLPNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class MLPNoShow:
    """
    MLP classifier for no-show prediction
    PyTorch-based neural network optimized for imbalanced data
    """
    
    def __init__(self, **params):
        """
        Initialize MLP model
        
        Args:
            **params: Model hyperparameters
        """
        # Default parameters
        self.params = {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 512,
            'epochs': 100,
            'weight_decay': 1e-4,
            **params
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = None
        
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                     fit_scaler: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data for PyTorch model
        
        Args:
            X: Features
            y: Target (optional)
            fit_scaler: Whether to fit the scaler
            
        Returns:
            X_tensor: Features tensor
            y_tensor: Target tensor (if provided)
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = None
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        return X_tensor, y_tensor
    
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
            feature_names: Optional feature names for importance
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
        """
        logger.info(f"Training MLP with params: {self.params}")
        logger.info(f"Device: {self.device}")
        
        self.feature_names = feature_names
        self.input_dim = X.shape[1]
        
        # Initialize model
        self.model = MLPNet(
            input_dim=self.input_dim,
            hidden_dims=self.params['hidden_dims'],
            dropout_rate=self.params['dropout_rate']
        ).to(self.device)
        
        # Prepare data
        X_train, y_train = self._prepare_data(X, y, fit_scaler=True)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True
        )
        
        # Prepare validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, fit_scaler=False)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
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
        logger.info("MLP training completed")
        
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
        X_tensor, _ = self._prepare_data(X, fit_scaler=False)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
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
        X_tensor, _ = self._prepare_data(X, fit_scaler=False)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        return probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (approximated using gradient-based method)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # For neural networks, we can't get traditional feature importance
        # Return dummy importance for consistency with other models
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.ones(len(self.feature_names)) / len(self.feature_names)
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(self.input_dim)],
                'importance': np.ones(self.input_dim) / self.input_dim
            }).sort_values('importance', ascending=False)
        
        logger.warning("MLP feature importance is uniform (neural networks don't provide traditional feature importance)")
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
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'input_dim': self.input_dim
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
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.is_fitted = save_dict['is_fitted']
        self.input_dim = save_dict['input_dim']
        
        if save_dict['model_state_dict'] and self.input_dim:
            self.model = MLPNet(
                input_dim=self.input_dim,
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
        # Optimize key hyperparameters for neural networks
        params = {}
        
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