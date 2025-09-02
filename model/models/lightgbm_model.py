"""
LightGBM model for no-show prediction
Optimized for imbalanced data with early stopping
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import joblib
import logging
from pathlib import Path

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from config.config import LGBM_DEFAULT_PARAMS, LGBM_SEARCH_SPACE, EARLY_STOPPING_ROUNDS

logger = logging.getLogger(__name__)


class LightGBMNoShow:
    """
    LightGBM classifier for no-show prediction
    Optimized for imbalanced data with early stopping
    """
    
    def __init__(self, **params):
        """
        Initialize LightGBM model
        
        Args:
            **params: Model hyperparameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            
        # Merge with default parameters
        self.params = {**LGBM_DEFAULT_PARAMS, **params}
        
        # Check if GPU is available through environment variable or auto-detection
        # Users with GPU can set environment variable: export LIGHTGBM_USE_GPU=1
        import os
        use_gpu = os.environ.get('LIGHTGBM_USE_GPU', '0') == '1'
        
        if use_gpu:
            # Add GPU parameters for users with proper GPU setup
            self.params['device'] = 'gpu'
            self.params['gpu_platform_id'] = 0
            self.params['gpu_device_id'] = 0
            logger.info("LightGBM GPU mode enabled (LIGHTGBM_USE_GPU=1)")
        else:
            # Remove any GPU parameters for CPU mode
            self.params.pop('device', None)
            self.params.pop('gpu_platform_id', None)
            self.params.pop('gpu_device_id', None)
            logger.info("LightGBM using CPU mode (set LIGHTGBM_USE_GPU=1 for GPU)")
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_names = None
        self.is_fitted = False
        
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
        logger.info(f"Training LightGBM with params: {self.params}")
        
        self.feature_names = feature_names
        
        # Training with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                eval_metric='average_precision',
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
            )
            logger.info(f"Early stopping at iteration {self.model.best_iteration_}")
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        logger.info("LightGBM training completed")
        
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
        
        return self.model.predict(X)
    
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
        
        # Return probability of positive class (no-show)
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.params = saved_data['params']
        self.feature_names = saved_data['feature_names']
        self.is_fitted = saved_data['is_fitted']
        logger.info(f"Model loaded from {filepath}")
    
    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        """
        Get Optuna search space for hyperparameter optimization
        
        Returns:
            Dictionary defining search space
        """
        return LGBM_SEARCH_SPACE
    
    @staticmethod
    def suggest_params(trial) -> Dict:
        """
        Suggest parameters for Optuna trial (optimized for efficiency)
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        # Only optimize key parameters for efficiency
        params['max_depth'] = trial.suggest_int('max_depth',
                                               LGBM_SEARCH_SPACE['max_depth']['low'],
                                               LGBM_SEARCH_SPACE['max_depth']['high'])
        
        params['num_leaves'] = trial.suggest_int('num_leaves',
                                               LGBM_SEARCH_SPACE['num_leaves']['low'],
                                               LGBM_SEARCH_SPACE['num_leaves']['high'])
        
        params['min_child_samples'] = trial.suggest_int('min_child_samples',
                                                       LGBM_SEARCH_SPACE['min_child_samples']['low'],
                                                       LGBM_SEARCH_SPACE['min_child_samples']['high'])
        
        return params
    
    @staticmethod
    def is_available() -> bool:
        """Check if LightGBM is available"""
        return LIGHTGBM_AVAILABLE