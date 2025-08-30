"""
TabNet model for no-show prediction
Sequential attention mechanism with feature selection for tabular data
Based on "TabNet: Attentive Interpretable Tabular Learning" paper
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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


class GLU(nn.Module):
    """Gated Linear Unit"""
    
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = None):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim * 2, momentum=0.02)
        self.virtual_batch_size = virtual_batch_size
        
    def forward(self, x):
        x = self.fc(x)
        
        if self.virtual_batch_size is not None:
            # Ghost batch normalization
            x = self.ghost_bn(x)
        else:
            x = self.bn(x)
        
        out, gate = x.chunk(2, dim=-1)
        return out * torch.sigmoid(gate)
    
    def ghost_bn(self, x):
        # Virtual batch normalization (simplified version)
        return self.bn(x)


class AttentiveTransformer(nn.Module):
    """Attentive Transformer for feature selection"""
    
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = None):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim, momentum=0.02)
        self.virtual_batch_size = virtual_batch_size
        
    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        
        if self.virtual_batch_size is not None:
            x = self.ghost_bn(x)
        else:
            x = self.bn(x)
        
        x = x * priors
        return torch.softmax(x, dim=-1)
    
    def ghost_bn(self, x):
        return self.bn(x)


class FeatureTransformer(nn.Module):
    """Feature Transformer block"""
    
    def __init__(self, input_dim: int, output_dim: int, shared_layers: nn.ModuleList,
                 n_glu_independent: int, virtual_batch_size: int = None):
        super().__init__()
        
        self.shared_layers = shared_layers
        self.specifics = nn.ModuleList()
        
        # Calculate input dimension for specific layers
        if shared_layers and len(shared_layers) > 0:
            spec_input_dim = 2 * (output_dim // 2)  # Shared layers output 2 * (n_d + n_a)
        else:
            spec_input_dim = input_dim
            
        for _ in range(n_glu_independent):
            self.specifics.append(GLU(spec_input_dim, output_dim, virtual_batch_size))
            spec_input_dim = output_dim
            
    def forward(self, x):
        # Shared layers
        for shared_layer in self.shared_layers:
            x = shared_layer(x)
            
        # Specific layers
        for specific_layer in self.specifics:
            x = specific_layer(x)
            
        return x


class TabNetNet(nn.Module):
    """
    TabNet Network with sequential attention
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, n_d: int = 8, n_a: int = 8,
                 n_steps: int = 3, gamma: float = 1.3, n_independent: int = 2, 
                 n_shared: int = 2, epsilon: float = 1e-15, virtual_batch_size: int = None,
                 momentum: float = 0.02):
        """
        Initialize TabNet network
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes
            n_d: Width of decision prediction layer
            n_a: Width of attention embedding
            n_steps: Number of decision steps
            gamma: Coefficient for feature reusage penalty
            n_independent: Number of independent GLU layers
            n_shared: Number of shared GLU layers  
            epsilon: Small constant for numerical stability
            virtual_batch_size: Size of virtual batches for ghost batch norm
            momentum: Momentum for batch normalization
        """
        super(TabNetNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
        # Shared layers for all decision steps
        if n_shared > 0:
            shared_feat_transform = nn.ModuleList()
            for i in range(n_shared):
                if i == 0:
                    shared_feat_transform.append(GLU(input_dim, 2 * (n_d + n_a), virtual_batch_size))
                else:
                    shared_feat_transform.append(GLU(2 * (n_d + n_a), 2 * (n_d + n_a), virtual_batch_size))
        else:
            shared_feat_transform = None
            
        # Decision step modules
        self.initial_splitter = FeatureTransformer(
            input_dim, 2 * (n_d + n_a), shared_feat_transform, n_independent, virtual_batch_size
        )
        
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        
        for step in range(n_steps):
            transformer = FeatureTransformer(
                input_dim, 2 * (n_d + n_a), shared_feat_transform, n_independent, virtual_batch_size
            )
            attention = AttentiveTransformer(n_a, input_dim, virtual_batch_size)
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)
        
        # Final classifier
        self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
        
    def forward(self, x):
        res = 0
        x = self.initial_bn(x)
        
        # Prior scales (initialized to 1)
        prior = torch.ones(x.shape).to(x.device)
        M_loss = 0
        att_list = []
        
        for step in range(self.n_steps):
            # Feature transformer
            M = self.initial_splitter(x * prior) if step == 0 else self.feat_transformers[step-1](x * prior)
            
            # Split into decision and attention parts
            M_feature, M_att = M[:, :self.n_d], M[:, self.n_d:self.n_d+self.n_a]
            
            # Attention
            mask = self.att_transformers[step](prior, M_att)
            att_list.append(mask)
            
            # Update prior for next step
            prior = prior * (self.gamma - mask)
            
            # Decision
            res = res + M_feature
            
            # Feature reusage penalty
            M_loss += torch.mean(torch.sum(torch.mul(mask, prior), dim=1))
        
        # Final prediction
        res = self.final_mapping(res)
        
        return res, M_loss, att_list
    
    def forward_masks(self, x):
        """Forward pass returning attention masks for interpretability"""
        _, _, att_list = self.forward(x)
        return torch.stack(att_list, dim=0)


class TabNetNoShow:
    """
    TabNet classifier for no-show prediction
    Sequential attention mechanism with feature selection
    """
    
    def __init__(self, **params):
        """
        Initialize TabNet model
        
        Args:
            **params: Model hyperparameters
        """
        # Default parameters
        self.params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-3,
            'learning_rate': 2e-2,
            'batch_size': 1024,
            'epochs': 100,
            'weight_decay': 1e-4,
            'virtual_batch_size': 512,
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
            feature_names: Optional feature names
            X_val: Validation features (for early stopping)
            y_val: Validation target (for early stopping)
        """
        logger.info(f"Training TabNet with params: {self.params}")
        logger.info(f"Device: {self.device}")
        
        self.feature_names = feature_names
        self.input_dim = X.shape[1]
        
        # Initialize model
        self.model = TabNetNet(
            input_dim=self.input_dim,
            output_dim=1,
            n_d=self.params['n_d'],
            n_a=self.params['n_a'],
            n_steps=self.params['n_steps'],
            gamma=self.params['gamma'],
            n_independent=self.params['n_independent'],
            n_shared=self.params['n_shared'],
            virtual_batch_size=self.params.get('virtual_batch_size')
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
                outputs, M_loss, _ = self.model(batch_X)
                
                # Main loss + sparsity regularization
                loss = criterion(outputs, batch_y) + self.params['lambda_sparse'] * M_loss
                
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
                        outputs, M_loss, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y) + self.params['lambda_sparse'] * M_loss
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
        logger.info("TabNet training completed")
        
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
            outputs, _, _ = self.model(X_tensor)
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
            outputs, _, _ = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        return probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using attention masks
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Get attention masks on a sample of training data
        # For simplicity, we'll create dummy importance based on the first attention layer
        # In practice, you might want to average over multiple samples
        
        if self.feature_names:
            # Create dummy importance for now (TabNet's actual importance requires running on data)
            importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        else:
            importance_scores = np.ones(self.input_dim) / self.input_dim
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(self.input_dim)],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        logger.warning("TabNet feature importance shown as uniform. For actual attention-based importance, use get_attention_masks() method.")
        return importance_df
    
    def get_attention_masks(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention masks for interpretability
        
        Args:
            X: Features to get attention for
            
        Returns:
            Attention masks array of shape (n_steps, n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting attention masks")
        
        self.model.eval()
        X_tensor, _ = self._prepare_data(X, fit_scaler=False)
        
        with torch.no_grad():
            masks = self.model.forward_masks(X_tensor)
        
        return masks.cpu().numpy()
    
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
            self.model = TabNetNet(
                input_dim=self.input_dim,
                output_dim=1,
                n_d=self.params['n_d'],
                n_a=self.params['n_a'],
                n_steps=self.params['n_steps'],
                gamma=self.params['gamma'],
                n_independent=self.params['n_independent'],
                n_shared=self.params['n_shared'],
                virtual_batch_size=self.params.get('virtual_batch_size')
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
        # Optimize key hyperparameters for TabNet
        params = {}
        
        # Decision and attention dimensions
        n_d = trial.suggest_categorical('n_d', [8, 16, 24])
        n_a = trial.suggest_categorical('n_a', [8, 16, 24])
        params['n_d'] = n_d
        params['n_a'] = n_a
        
        # Number of decision steps
        params['n_steps'] = trial.suggest_int('n_steps', 3, 6)
        
        # Feature reusage penalty
        params['gamma'] = trial.suggest_float('gamma', 1.0, 1.5)
        
        # Architecture parameters
        params['n_independent'] = trial.suggest_int('n_independent', 1, 3)
        params['n_shared'] = trial.suggest_int('n_shared', 1, 3)
        
        # Regularization
        params['lambda_sparse'] = trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True)
        
        # Training parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 5e-3, 5e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [512, 1024, 2048])
        
        return params
    
    @staticmethod
    def is_available() -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False