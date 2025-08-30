"""
Hyperparameter optimization using Optuna
For systematic hyperparameter tuning in research
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
import logging
import time
from sklearn.metrics import average_precision_score

from training.validator import TimeSeriesValidator
from evaluation.metrics import MetricsCalculator
from models.random_forest import RandomForestNoShow
from config.config import OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, RANDOM_SEED, CV_FOLDS

logger = logging.getLogger(__name__)

# Suppress Optuna logs except warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna with TimeSeriesSplit
    Optimizes for PR-AUC (best for imbalanced data)
    """
    
    def __init__(self, model_class, X: np.ndarray, y: np.ndarray,
                 cv_splitter: Optional[TimeSeriesValidator] = None,
                 metric: str = 'pr_auc',
                 n_trials: int = OPTUNA_N_TRIALS,
                 timeout: Optional[float] = OPTUNA_TIMEOUT,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize optimizer
        
        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            cv_splitter: Cross-validation splitter
            metric: Metric to optimize ('pr_auc', 'f2', 'roc_auc')
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            random_seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.X = X
        self.y = y
        self.cv_splitter = cv_splitter or TimeSeriesValidator(n_splits=CV_FOLDS)
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
        
        # Storage for trial results
        self.trial_results = []
        self.best_params = None
        self.best_score = None
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
    def objective(self, trial) -> float:
        """
        Objective function for Optuna
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean CV score for the metric
        """
        # Suggest parameters
        params = self.model_class.suggest_params(trial)
        params['random_state'] = self.random_seed
        params['n_jobs'] = -1  # Use all cores
        
        # Cross-validation scores
        cv_scores = []
        fold_times = []
        
        # Time series cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv_splitter.split(self.X, self.y), 1):
            # Split data
            X_train_fold = self.X[train_idx]
            y_train_fold = self.y[train_idx]
            X_val_fold = self.X[val_idx]
            y_val_fold = self.y[val_idx]
            
            # Train model
            model = self.model_class(**params)
            
            start_time = time.time()
            model.fit(X_train_fold, y_train_fold)
            train_time = time.time() - start_time
            fold_times.append(train_time)
            
            # Predict
            y_prob = model.predict_proba(X_val_fold)
            
            # Calculate metric
            if self.metric == 'pr_auc':
                score = average_precision_score(y_val_fold, y_prob)
            elif self.metric == 'f2':
                # Find optimal threshold and calculate F2
                y_pred = (y_prob >= 0.5).astype(int)
                score = self.metrics_calc._calculate_fbeta(y_val_fold, y_pred, beta=2.0)
            elif self.metric == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val_fold, y_prob)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            cv_scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold - 1)
            
            # Handle pruning
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at fold {fold}")
                raise optuna.TrialPruned()
        
        # Calculate mean score
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        mean_time = np.mean(fold_times)
        
        # Store trial results
        trial_result = {
            'trial': trial.number,
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_scores': cv_scores,
            'mean_time': mean_time
        }
        self.trial_results.append(trial_result)
        
        # Log progress
        if trial.number % 10 == 0:
            logger.info(f"Trial {trial.number}/{self.n_trials} | {self.metric}: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score
    
    def optimize(self, show_progress: bool = True) -> Dict:
        """
        Run optimization
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with best parameters and optimization history
        """
        logger.info(f"Starting Optuna optimization for {self.n_trials} trials")
        logger.info(f"Optimizing {self.metric} with {CV_FOLDS}-fold TimeSeriesSplit")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
        )
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=show_progress,
            n_jobs=1  # Sequential for reproducibility
        )
        
        optimization_time = time.time() - start_time
        
        # Get best results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Create results summary
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_time': optimization_time,
            'n_trials': len(study.trials),
            'study': study,
            'trial_history': self.get_trial_history()
        }
        
        return results
    
    def get_trial_history(self) -> pd.DataFrame:
        """
        Get trial history as DataFrame
        
        Returns:
            DataFrame with trial results
        """
        if not self.trial_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.trial_results)
        
        # Add rank
        df['rank'] = df['mean_score'].rank(ascending=False, method='min')
        
        # Sort by score
        df = df.sort_values('mean_score', ascending=False)
        
        return df
    
    def get_param_importance(self, study: optuna.Study) -> pd.DataFrame:
        """
        Get parameter importance from study
        
        Args:
            study: Optuna study object
            
        Returns:
            DataFrame with parameter importance
        """
        try:
            importance = optuna.importance.get_param_importances(study)
            
            importance_df = pd.DataFrame([
                {'parameter': k, 'importance': v}
                for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
            
            return importance_df
        except:
            logger.warning("Could not calculate parameter importance")
            return pd.DataFrame()
    
    def plot_optimization_history(self, study: optuna.Study):
        """
        Plot optimization history (for notebook environments)
        
        Args:
            study: Optuna study object
        """
        try:
            import optuna.visualization as vis
            
            # Optimization history
            fig1 = vis.plot_optimization_history(study)
            
            # Parameter importance
            fig2 = vis.plot_param_importances(study)
            
            # Parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(study)
            
            return fig1, fig2, fig3
        except:
            logger.warning("Could not create optimization plots")
            return None, None, None