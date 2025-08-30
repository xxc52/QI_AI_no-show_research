"""
Main orchestrator for no-show prediction models
Research pipeline for paper results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import logging.config
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config.config import (
    LOGGING_CONFIG, RESULTS_ROOT, RANDOM_SEED,
    CV_FOLDS, OPTUNA_N_TRIALS
)
from data.loader import DataLoader
from training.validator import TimeSeriesValidator
from training.optimizer import OptunaOptimizer
from models.random_forest import RandomForestNoShow
from models.lightgbm_model import LightGBMNoShow
from models.mlp_model import MLPNoShow
from models.deepfm_model import DeepFMNoShow
from models.fttransformer_model import FTTransformerNoShow
from models.tabnet_model import TabNetNoShow
from evaluation.metrics import MetricsCalculator
from utils.timer import Timer, measure_training_time, measure_prediction_time, format_time

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class NoShowPipeline:
    """Main pipeline for no-show prediction research"""
    
    def __init__(self, model_type: str = 'randomforest', random_seed: int = RANDOM_SEED):
        """
        Initialize pipeline
        
        Args:
            model_type: Type of model to use
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.cv_splitter = TimeSeriesValidator(n_splits=CV_FOLDS)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Pipeline initialized for {model_type} model")
        
    def run_complete_pipeline(self, optimize: bool = True, n_trials: int = OPTUNA_N_TRIALS):
        """
        Run complete pipeline: optimization, training, evaluation
        
        Args:
            optimize: Whether to run hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Dictionary with all results
        """
        logger.info("="*70)
        logger.info("Starting No-Show Prediction Pipeline")
        logger.info("="*70)
        
        # Step 1: Load Data
        logger.info("\n[Step 1/5] Loading Data...")
        with Timer("Data Loading"):
            X_train, y_train, X_test, y_test, train_ids, test_ids = self.data_loader.get_train_test_split()
            feature_names = self.data_loader.get_feature_names()
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        # Step 2: Hyperparameter Optimization
        best_params = {}
        if optimize:
            logger.info(f"\n[Step 2/5] Hyperparameter Optimization ({n_trials} trials)...")
            with Timer("Hyperparameter Optimization"):
                best_params = self.optimize_hyperparameters(X_train, y_train, n_trials)
        else:
            logger.info("\n[Step 2/5] Skipping optimization, using default parameters")
            best_params = {}
        
        # Step 3: Train Final Model
        logger.info("\n[Step 3/5] Training Final Model...")
        model, train_time = self.train_final_model(X_train, y_train, best_params, feature_names)
        
        # Step 4: Evaluate on Test Set
        logger.info("\n[Step 4/5] Evaluating on Test Set...")
        test_metrics = self.evaluate_model(model, X_test, y_test, train_time)
        
        # Step 5: Save Results
        logger.info("\n[Step 5/5] Saving Results...")
        self.save_results(model, test_metrics, best_params)
        
        # Print final summary
        self.print_summary(test_metrics)
        
        return {
            'model': model,
            'metrics': test_metrics,
            'best_params': best_params,
            'feature_importance': model.get_feature_importance()
        }
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                n_trials: int = OPTUNA_N_TRIALS) -> dict:
        """
        Run hyperparameter optimization
        
        Args:
            X: Training features
            y: Training target
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters dictionary
        """
        if self.model_type == 'randomforest':
            model_class = RandomForestNoShow
        elif self.model_type == 'lightgbm':
            if not LightGBMNoShow.is_available():
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            model_class = LightGBMNoShow
        elif self.model_type == 'mlp':
            if not MLPNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model_class = MLPNoShow
        elif self.model_type == 'deepfm':
            if not DeepFMNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model_class = DeepFMNoShow
        elif self.model_type == 'fttransformer':
            if not FTTransformerNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model_class = FTTransformerNoShow
        elif self.model_type == 'tabnet':
            if not TabNetNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model_class = TabNetNoShow
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        optimizer = OptunaOptimizer(
            model_class=model_class,
            X=X,
            y=y,
            cv_splitter=self.cv_splitter,
            metric='pr_auc',  # Best for imbalanced data
            n_trials=n_trials,
            random_seed=self.random_seed
        )
        
        results = optimizer.optimize(show_progress=True)
        
        # Store optimization results
        self.results['optimization'] = results
        
        # Save trial history
        trial_history = results['trial_history']
        if not trial_history.empty:
            trial_history.to_csv(RESULTS_ROOT / 'logs' / f'{self.model_type}_trials.csv', index=False)
            logger.info(f"Trial history saved to {self.model_type}_trials.csv")
        
        return results['best_params']
    
    def train_final_model(self, X: np.ndarray, y: np.ndarray, 
                         params: dict, feature_names: list):
        """
        Train final model with best parameters
        
        Args:
            X: Training features
            y: Training target
            params: Model parameters
            feature_names: Feature names
            
        Returns:
            Tuple of (trained_model, training_time)
        """
        if self.model_type == 'randomforest':
            model = RandomForestNoShow(**params)
        elif self.model_type == 'lightgbm':
            if not LightGBMNoShow.is_available():
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            model = LightGBMNoShow(**params)
        elif self.model_type == 'mlp':
            if not MLPNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model = MLPNoShow(**params)
        elif self.model_type == 'deepfm':
            if not DeepFMNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model = DeepFMNoShow(**params)
        elif self.model_type == 'fttransformer':
            if not FTTransformerNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model = FTTransformerNoShow(**params)
        elif self.model_type == 'tabnet':
            if not TabNetNoShow.is_available():
                raise ImportError("PyTorch not available. Install with: pip install torch")
            model = TabNetNoShow(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Measure training time
        train_time = measure_training_time(model, X, y)
        
        # Set feature names
        model.feature_names = feature_names
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        logger.info("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string())
        
        return model, train_time
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      train_time: float) -> dict:
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            train_time: Training time in seconds
            
        Returns:
            Dictionary with all metrics
        """
        # Make predictions
        with Timer("Prediction"):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        
        # Measure prediction time
        pred_total_time, pred_time_ms, pred_std_ms = measure_prediction_time(model, X_test, n_runs=10)
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_test, y_pred, y_prob, include_curves=True
        )
        
        # Add timing metrics
        metrics['train_time'] = train_time
        metrics['pred_time_total'] = pred_total_time
        metrics['pred_time_ms'] = pred_time_ms
        metrics['pred_time_std_ms'] = pred_std_ms
        metrics['n_test_samples'] = len(X_test)
        
        return metrics
    
    def save_results(self, model, metrics: dict, params: dict):
        """
        Save all results to disk
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            params: Model parameters
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = RESULTS_ROOT / 'models' / f'{self.model_type}_{timestamp}.pkl'
        model.save(str(model_path))
        
        # Save metrics
        metrics_path = RESULTS_ROOT / 'logs' / f'{self.model_type}_metrics_{timestamp}.json'
        
        # Convert numpy arrays and special types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_json[key] = float(value)
            elif key in ['roc_curve', 'pr_curve']:
                # Skip curve data for JSON
                continue
            else:
                metrics_json[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'parameters': params,
                'metrics': metrics_json,
                'timestamp': timestamp
            }, f, indent=2)
        
        logger.info(f"Results saved to {RESULTS_ROOT}")
        
    def print_summary(self, metrics: dict):
        """
        Print formatted summary of results
        
        Args:
            metrics: Dictionary with metrics
        """
        # Use the metrics formatter
        summary = self.metrics_calculator.format_metrics_table(metrics, self.model_type.upper())
        print("\n" + summary)
        
        # Additional paper-specific table
        print("\n" + "="*80)
        print("PAPER RESULTS TABLE")
        print("="*80)
        
        table_data = {
            'Model': self.model_type.upper(),
            'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
            'PR-AUC': f"{metrics.get('pr_auc', 0):.4f}",
            'F1': f"{metrics.get('f1', 0):.4f}",
            'F2': f"{metrics.get('f2', 0):.4f}",
            'MCC': f"{metrics.get('mcc', 0):.4f}",
            'Brier': f"{metrics.get('brier_score', 0):.4f}",
            'G-Mean': f"{metrics.get('g_mean', 0):.4f}",
            'Train(s)': f"{metrics.get('train_time', 0):.1f}",
            'Pred(ms)': f"{metrics.get('pred_time_ms', 0):.2f}"
        }
        
        # Print as formatted table
        df = pd.DataFrame([table_data])
        print(df.to_string(index=False))
        print("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='No-Show Prediction Pipeline')
    
    parser.add_argument('--model', type=str, default='randomforest',
                       choices=['randomforest', 'lightgbm', 'mlp', 'deepfm', 'fttransformer', 'tabnet', 'all'],
                       help='Model type to use (use "all" for comprehensive benchmark)')
    
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Run hyperparameter optimization')
    
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                       help='Skip hyperparameter optimization')
    
    parser.add_argument('--trials', type=int, default=OPTUNA_N_TRIALS,
                       help='Number of Optuna trials')
    
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Handle 'all' models - run comprehensive benchmark
    if args.model == 'all':
        from benchmark import ModelBenchmark
        
        logger.info("Running comprehensive benchmark on all models...")
        benchmark = ModelBenchmark(n_trials=args.trials, random_seed=args.seed)
        results = benchmark.run_benchmark(optimize=args.optimize)
        logger.info("\nComprehensive benchmark completed!")
        return
    
    # Create results directories
    (RESULTS_ROOT / 'models').mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    
    # Run single model pipeline
    pipeline = NoShowPipeline(model_type=args.model, random_seed=args.seed)
    results = pipeline.run_complete_pipeline(optimize=args.optimize, n_trials=args.trials)
    
    logger.info("\nPipeline completed successfully!")
    

if __name__ == "__main__":
    main()