"""
FT-Transformer model for no-show prediction
Feature Tokenizer + Transformer architecture for tabular data
Based on "Revisiting Deep Learning Models for Tabular Data" paper
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
import math

from config.config import RANDOM_SEED, EARLY_STOPPING_ROUNDS

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Store residual
        residual = x
        
        # Apply layer norm (pre-norm)
        x = self.layer_norm(x)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attention_output)
        
        # Add residual connection
        return output + residual


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Store residual
        residual = x
        
        # Apply layer norm (pre-norm)
        x = self.layer_norm(x)
        
        # Feed forward
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        # Add residual connection
        return x + residual


class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Head Attention and Feed-Forward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class FTTransformerNet(nn.Module):
    """
    FT-Transformer Network for tabular data
    """
    
    def __init__(self, num_features: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize FT-Transformer network
        
        Args:
            num_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(FTTransformerNet, self).__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        
        # Feature tokenization - each feature gets its own learnable token
        self.feature_tokenizers = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_features + 1, d_model))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Tokenize each feature
        tokens = []
        for i in range(self.num_features):
            feature_val = x[:, i:i+1]  # Shape: (batch_size, 1)
            token = self.feature_tokenizers[i](feature_val)  # Shape: (batch_size, d_model)
            tokens.append(token.unsqueeze(1))  # Shape: (batch_size, 1, d_model)
        
        # Concatenate feature tokens
        feature_tokens = torch.cat(tokens, dim=1)  # Shape: (batch_size, num_features, d_model)
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, feature_tokens], dim=1)  # Shape: (batch_size, num_features+1, d_model)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            tokens = transformer_block(tokens)
        
        # Apply final layer norm
        tokens = self.layer_norm(tokens)
        
        # Use CLS token for classification
        cls_output = tokens[:, 0]  # Shape: (batch_size, d_model)
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


class FTTransformerNoShow:
    """
    FT-Transformer classifier for no-show prediction
    Feature Tokenizer + Transformer architecture for tabular data
    """
    
    def __init__(self, **params):
        """
        Initialize FT-Transformer model
        
        Args:
            **params: Model hyperparameters
        """
        # Default parameters
        self.params = {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 3,
            'd_ff': 256,
            'dropout_rate': 0.1,
            'learning_rate': 1e-4,
            'batch_size': 256,
            'epochs': 100,
            'weight_decay': 1e-4,
            **params
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = None
        
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
            feature_names: Optional feature names
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
        """
        logger.info(f"Training FT-Transformer with params: {self.params}")
        logger.info(f"Device: {self.device}")
        
        self.feature_names = feature_names
        self.num_features = X.shape[1]
        
        # Initialize model
        self.model = FTTransformerNet(
            num_features=self.num_features,
            d_model=self.params['d_model'],
            num_heads=self.params['num_heads'],
            num_layers=self.params['num_layers'],
            d_ff=self.params['d_ff'],
            dropout=self.params['dropout_rate']
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
        optimizer = optim.AdamW(
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
        logger.info("FT-Transformer training completed")
        
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
        Get feature importance (approximated using attention weights)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # For Transformer, we can't get traditional feature importance
        # Return dummy importance for consistency with other models
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.ones(len(self.feature_names)) / len(self.feature_names)
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(self.num_features)],
                'importance': np.ones(self.num_features) / self.num_features
            }).sort_values('importance', ascending=False)
        
        logger.warning("FT-Transformer feature importance is uniform (transformer attention doesn't directly translate to feature importance)")
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
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.is_fitted = save_dict['is_fitted']
        self.num_features = save_dict['num_features']
        
        if save_dict['model_state_dict'] and self.num_features:
            self.model = FTTransformerNet(
                num_features=self.num_features,
                d_model=self.params['d_model'],
                num_heads=self.params['num_heads'],
                num_layers=self.params['num_layers'],
                d_ff=self.params['d_ff'],
                dropout=self.params['dropout_rate']
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
        # Optimize key hyperparameters for FT-Transformer
        params = {}
        
        # Model dimension (must be divisible by num_heads)
        d_model_choice = trial.suggest_categorical('d_model', [64, 128, 192])
        params['d_model'] = d_model_choice
        
        # Number of attention heads (must divide d_model evenly)
        if d_model_choice == 64:
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8])
        elif d_model_choice == 128:
            params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8])
        else:  # 192
            params['num_heads'] = trial.suggest_categorical('num_heads', [6, 8, 12])
        
        # Number of transformer layers
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        
        # Feed-forward dimension (usually 2-4x d_model)
        ff_multiplier = trial.suggest_categorical('ff_multiplier', [2, 3, 4])
        params['d_ff'] = d_model_choice * ff_multiplier
        
        # Key hyperparameters
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.3)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        return params
    
    @staticmethod
    def is_available() -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False