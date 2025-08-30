"""
Time Series Cross-Validation for temporal data
Ensures temporal integrity and prevents data leakage
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Time series cross-validation with patient-aware splitting
    Ensures no data leakage in temporal medical data
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[float] = None):
        """
        Initialize time series validator
        
        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set (as fraction of total)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.base_splitter = SklearnTimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
    def split(self, X: np.ndarray, y: np.ndarray, 
              patient_ids: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation splits maintaining temporal order
        
        Args:
            X: Feature matrix
            y: Target vector
            patient_ids: Patient IDs for leak prevention (optional)
            
        Yields:
            train_idx: Training indices
            val_idx: Validation indices
        """
        logger.info(f"Starting TimeSeriesSplit with {self.n_splits} folds")
        
        # If no patient IDs provided, use standard time series split
        if patient_ids is None:
            for fold, (train_idx, val_idx) in enumerate(self.base_splitter.split(X), 1):
                logger.info(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}")
                yield train_idx, val_idx
        else:
            # Patient-aware splitting to prevent leakage
            for fold, (train_idx, val_idx) in enumerate(self.base_splitter.split(X), 1):
                # Ensure no patient appears in both train and validation
                train_patients = set(patient_ids[train_idx])
                val_patients = set(patient_ids[val_idx])
                
                # Check for patient overlap
                overlap = train_patients & val_patients
                if overlap:
                    logger.warning(f"Fold {fold}: Found {len(overlap)} overlapping patients")
                    # Remove overlapping patients from validation
                    val_idx = val_idx[~np.isin(patient_ids[val_idx], list(overlap))]
                
                logger.info(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)} (after patient dedup)")
                yield train_idx, val_idx
    
    def get_fold_statistics(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Get statistics for each fold
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with fold statistics
        """
        stats = []
        
        for fold, (train_idx, val_idx) in enumerate(self.base_splitter.split(X), 1):
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            fold_stats = {
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_pos_rate': np.mean(y_train_fold),
                'val_pos_rate': np.mean(y_val_fold),
                'train_pos_count': np.sum(y_train_fold),
                'val_pos_count': np.sum(y_val_fold)
            }
            stats.append(fold_stats)
        
        return pd.DataFrame(stats)
    
    def validate_temporal_order(self, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
        """
        Validate that validation indices come after training indices
        
        Args:
            train_idx: Training indices
            val_idx: Validation indices
            
        Returns:
            True if temporal order is maintained
        """
        return np.max(train_idx) < np.min(val_idx)


class BlockTimeSeriesValidator:
    """
    Block time series validation for contiguous time periods
    Useful for maintaining temporal blocks in medical data
    """
    
    def __init__(self, n_splits: int = 5, min_train_size: Optional[int] = None):
        """
        Initialize block time series validator
        
        Args:
            n_splits: Number of splits
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        
    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation splits with expanding window
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Yields:
            train_idx: Training indices
            val_idx: Validation indices
        """
        n_samples = len(X)
        
        # Calculate split points
        if self.min_train_size is None:
            self.min_train_size = n_samples // (self.n_splits + 1)
        
        # Generate splits
        for i in range(self.n_splits):
            # Expanding window for training
            train_end = self.min_train_size + i * (n_samples - self.min_train_size) // self.n_splits
            val_end = self.min_train_size + (i + 1) * (n_samples - self.min_train_size) // self.n_splits
            
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, min(val_end, n_samples))
            
            logger.info(f"Block {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
            
            yield train_idx, val_idx