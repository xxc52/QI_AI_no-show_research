"""
Data loading utilities for no-show prediction
Handles temporal split data and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional

from config.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    SELECTED_FEATURES, TARGET_COLUMN, ID_COLUMNS,
    RANDOM_SEED
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for no-show prediction models"""
    
    def __init__(self, selected_features: Optional[List[str]] = None):
        """
        Initialize data loader
        
        Args:
            selected_features: List of features to use. If None, uses config defaults
        """
        self.selected_features = selected_features or SELECTED_FEATURES
        self.target_column = TARGET_COLUMN
        self.id_columns = ID_COLUMNS
        
    def load_temporal_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load pre-split temporal data
        
        Returns:
            train_df: Training data
            val_df: Validation data  
            test_df: Test data
        """
        logger.info("Loading temporal split data...")
        
        # Load data
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        logger.info(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Verify columns exist
        self._verify_columns(train_df)
        
        return train_df, val_df, test_df
    
    def load_train_val_combined(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine train + validation for final training
        
        Returns:
            combined_df: Combined train+val data
            test_df: Test data
        """
        train_df, val_df, test_df = self.load_temporal_split_data()
        
        # Combine train and validation
        combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        
        # Sort by date to maintain temporal order
        if 'Appointment_Date' in combined_df.columns:
            combined_df = combined_df.sort_values('Appointment_Date')
        
        logger.info(f"Combined train+val: {len(combined_df)} samples")
        
        return combined_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Prepare features and target for modeling
        
        Args:
            df: Input dataframe
            
        Returns:
            X: Feature matrix
            y: Target vector
            ids: DataFrame with ID columns (PatientId, AppointmentID)
        """
        # Keep ID columns separate (for data leakage prevention)
        id_cols_present = [col for col in self.id_columns if col in df.columns]
        ids = df[id_cols_present].copy() if id_cols_present else pd.DataFrame()
        
        # Extract features and target
        feature_cols = [col for col in self.selected_features if col in df.columns]
        missing_features = set(self.selected_features) - set(feature_cols)
        
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}")
        
        X = df[feature_cols].copy()
        
        # Handle target column
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        y = df[self.target_column].copy()
        
        # Convert target to binary (if string)
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)
        
        logger.info(f"Prepared features: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, ids
    
    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Get train+val (combined) and test sets ready for modeling
        
        Returns:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            train_ids: Training IDs
            test_ids: Test IDs
        """
        # Load combined train+val and test
        train_df, test_df = self.load_train_val_combined()
        
        # Prepare features
        X_train, y_train, train_ids = self.prepare_features(train_df)
        X_test, y_test, test_ids = self.prepare_features(test_df)
        
        # Convert to numpy arrays for scikit-learn
        X_train = X_train.values
        y_train = y_train.values
        X_test = X_test.values
        y_test = y_test.values
        
        logger.info(f"Final split - Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test, train_ids, test_ids
    
    def _verify_columns(self, df: pd.DataFrame) -> None:
        """Verify required columns exist in dataframe"""
        missing_features = set(self.selected_features) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        missing_ids = set(self.id_columns) - set(df.columns)
        if missing_ids:
            logger.warning(f"Missing ID columns: {missing_ids}")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.selected_features
    
    def get_class_weights(self, y: np.ndarray) -> dict:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y: Target array
            
        Returns:
            Class weight dictionary
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weights}")
        
        return class_weights