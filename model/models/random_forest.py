"""
RandomForest model for no-show prediction
With hyperparameter optimization via Optuna
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Optional, Any
import joblib
import logging
from pathlib import Path

from config.config import RF_DEFAULT_PARAMS, RF_SEARCH_SPACE

logger = logging.getLogger(__name__)


class RandomForestNoShow:
    """
    RandomForest classifier for no-show prediction
    Optimized for imbalanced data
    """
    
    def __init__(self, **params):
        """
        Initialize RandomForest model
        
        Args:
            **params: Model hyperparameters
        """
        # Merge with default parameters
        self.params = {**RF_DEFAULT_PARAMS, **params}
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Train the model
        
        Args:
            X: Training features
            y: Training target
            feature_names: Optional feature names for importance
        """
        logger.info(f"Training RandomForest with params: {self.params}")
        
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("RandomForest training completed")
        
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
        return RF_SEARCH_SPACE
    
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
                                               RF_SEARCH_SPACE['max_depth']['low'],
                                               RF_SEARCH_SPACE['max_depth']['high'],
                                               step=RF_SEARCH_SPACE['max_depth']['step'])
        
        params['min_samples_split'] = trial.suggest_int('min_samples_split',
                                                       RF_SEARCH_SPACE['min_samples_split']['low'],
                                                       RF_SEARCH_SPACE['min_samples_split']['high'])
        
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf',
                                                      RF_SEARCH_SPACE['min_samples_leaf']['low'],
                                                      RF_SEARCH_SPACE['min_samples_leaf']['high'])
        
        return params