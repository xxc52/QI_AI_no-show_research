"""
Configuration for No-Show Prediction Models
Research-focused configuration for paper results
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Data paths
TRAIN_PATH = DATA_ROOT / "train.csv"
VAL_PATH = DATA_ROOT / "val.csv"
TEST_PATH = DATA_ROOT / "test.csv"

# Feature selection results path
FEATURE_SELECTION_PATH = PROJECT_ROOT.parent / "feature_selection_analysis"
SELECTED_FEATURES_PATH = FEATURE_SELECTION_PATH / "ml_dataset_selected_features.csv"

# Selected features from univariate statistical test (24 features)
SELECTED_FEATURES = [
    'lead_time', 'is_same_day', 'Registration_Hour', 'Registration_Month',
    'Registration_Day', 'Registration_Shift',
    'patient_previous_noshow_count', 'patient_appointment_count',
    'patient_previous_noshow_rate', 'days_since_last_appointment',
    'time_between_appointments_avg', 'appointment_regularity',
    'Hipertension', 'Diabetes', 'Handcap',
    'Age', 'Scholarship',
    'SMS_received',
    'neighbourhood_cluster_encoded',
    'temp_change', 'rad_max', 'temp_range', 'temp_min',
    'season_encoded'
]

# Target variable
TARGET_COLUMN = 'No-show'

# ID columns (not features but needed for data leakage prevention)
ID_COLUMNS = ['PatientId', 'AppointmentID']

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation settings
CV_FOLDS = 5
CV_TEST_SIZE = 0.2  # For TimeSeriesSplit

# Optuna settings
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = None  # No timeout for research
OPTUNA_N_JOBS = 1  # Sequential for reproducibility in research

# Model training settings
EARLY_STOPPING_ROUNDS = 150  # For gradient boosting models (100-200 range, using middle value)
VERBOSE = True  # Detailed output for research

# Metrics to track (comprehensive for imbalanced data)
METRICS_CONFIG = {
    'basic': ['accuracy', 'precision', 'recall', 'f1'],
    'imbalanced': ['f2', 'mcc', 'cohen_kappa', 'balanced_accuracy'],
    'probabilistic': ['roc_auc', 'pr_auc', 'brier_score', 'log_loss'],
    'threshold_dependent': ['gmean', 'youdens_j'],
    'top_k': [10, 20, 30],  # Precision/Recall @ Top K%
}

# RandomForest default parameters
RF_DEFAULT_PARAMS = {
    'n_estimators': 1000,  # Fixed high value for stability
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',  # Fixed for efficiency
    'criterion': 'gini',  # Fixed for efficiency
    'class_weight': 'balanced',  # Fixed for imbalanced data
    'random_state': RANDOM_SEED,
    'n_jobs': -1,  # Use all cores
    'verbose': 0
}

# RandomForest search space for Optuna (optimized for efficiency - only 3 core parameters)
RF_SEARCH_SPACE = {
    'max_depth': {'type': 'categorical', 'choices': [None, 15]},  # None vs 15 as requested
    'min_samples_leaf': {'type': 'categorical', 'choices': [1, 5]},  # 1 vs 5 as requested
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}  # Third parameter to optimize
}

# LightGBM default parameters (updated based on user feedback)
LGBM_DEFAULT_PARAMS = {
    'n_estimators': 2000,  # Increased for better performance
    'max_depth': -1,
    'num_leaves': 31,
    'min_child_samples': 100,  # Updated from 20 to 100 as requested
    'learning_rate': 0.05,  # Reduced learning rate as requested
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,  # Updated from 5 to 1 as requested
    'objective': 'binary',  # Fixed for binary classification
    'metric': 'average_precision',  # AUC-PR (same as Optuna optimization metric)
    'scale_pos_weight': 4,  # Use scale_pos_weight=4 instead of class_weight='balanced'
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1  # Suppress warnings
}

# LightGBM search space for Optuna (expanded based on user feedback)  
LGBM_SEARCH_SPACE = {
    'max_depth': {'type': 'int', 'low': 6, 'high': 12},  # Expanded range 6-12
    'num_leaves': {'type': 'int', 'low': 31, 'high': 127},  # Expanded range 31-127  
    'min_child_samples': {'type': 'int', 'low': 50, 'high': 200},  # Expanded range 50-200
    'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.95},  # Added to search space
    'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 0.95},  # Added to search space
    'min_split_gain': {'type': 'float', 'low': 0.0, 'high': 0.1},  # Added to search space
    'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 10.0},  # Added to search space
    'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 10.0}   # Added to search space
}

# Neural Network Models Default Parameters and Search Spaces

# MLP default parameters
MLP_DEFAULT_PARAMS = {
    'hidden_dims': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 512,
    'epochs': 100,
    'weight_decay': 1e-4
}

# MLP search space for Optuna (optimized for efficiency)
MLP_SEARCH_SPACE = {
    'architecture': {'type': 'categorical', 'choices': ['small', 'medium', 'large']},
    'dropout_rate': {'type': 'float', 'low': 0.2, 'high': 0.5},
    'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024]}
}

# DeepFM default parameters
DEEPFM_DEFAULT_PARAMS = {
    'embedding_dim': 16,
    'hidden_dims': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 512,
    'epochs': 100,
    'weight_decay': 1e-4
}

# DeepFM search space for Optuna (optimized for efficiency)
DEEPFM_SEARCH_SPACE = {
    'embedding_dim': {'type': 'categorical', 'choices': [8, 16, 32]},
    'architecture': {'type': 'categorical', 'choices': ['small', 'medium', 'large']},
    'dropout_rate': {'type': 'float', 'low': 0.2, 'high': 0.5},
    'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024]}
}

# FT-Transformer default parameters
FTTRANSFORMER_DEFAULT_PARAMS = {
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 3,
    'd_ff': 256,
    'dropout_rate': 0.1,
    'learning_rate': 1e-4,
    'batch_size': 256,
    'epochs': 100,
    'weight_decay': 1e-4
}

# FT-Transformer search space for Optuna (optimized for efficiency)
FTTRANSFORMER_SEARCH_SPACE = {
    'd_model': {'type': 'categorical', 'choices': [64, 128, 192]},
    'num_heads': {'type': 'conditional', 'choices': {64: [4, 8], 128: [4, 8], 192: [6, 8, 12]}},
    'num_layers': {'type': 'int', 'low': 2, 'high': 4},
    'ff_multiplier': {'type': 'categorical', 'choices': [2, 3, 4]},
    'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.3},
    'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [128, 256, 512]}
}

# TabNet default parameters
TABNET_DEFAULT_PARAMS = {
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
    'virtual_batch_size': 512
}

# TabNet search space for Optuna (optimized for efficiency)
TABNET_SEARCH_SPACE = {
    'n_d': {'type': 'categorical', 'choices': [8, 16, 24]},
    'n_a': {'type': 'categorical', 'choices': [8, 16, 24]},
    'n_steps': {'type': 'int', 'low': 3, 'high': 6},
    'gamma': {'type': 'float', 'low': 1.0, 'high': 1.5},
    'n_independent': {'type': 'int', 'low': 1, 'high': 3},
    'n_shared': {'type': 'int', 'low': 1, 'high': 3},
    'lambda_sparse': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    'learning_rate': {'type': 'float', 'low': 5e-3, 'high': 5e-2, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [512, 1024, 2048]}
}

# Centralized model configuration
MODEL_CONFIGS = {
    'randomforest': {
        'default_params': RF_DEFAULT_PARAMS,
        'search_space': RF_SEARCH_SPACE
    },
    'lightgbm': {
        'default_params': LGBM_DEFAULT_PARAMS,
        'search_space': LGBM_SEARCH_SPACE
    },
    'mlp': {
        'default_params': MLP_DEFAULT_PARAMS,
        'search_space': MLP_SEARCH_SPACE
    },
    'deepfm': {
        'default_params': DEEPFM_DEFAULT_PARAMS,
        'search_space': DEEPFM_SEARCH_SPACE
    },
    'fttransformer': {
        'default_params': FTTRANSFORMER_DEFAULT_PARAMS,
        'search_space': FTTRANSFORMER_SEARCH_SPACE
    },
    'tabnet': {
        'default_params': TABNET_DEFAULT_PARAMS,
        'search_space': TABNET_SEARCH_SPACE
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': str(RESULTS_ROOT / 'logs' / 'training.log')
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}