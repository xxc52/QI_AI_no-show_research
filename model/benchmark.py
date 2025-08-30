"""
Comprehensive model benchmarking for no-show prediction
Runs all models with optimization and generates comparison results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config.config import (
    RESULTS_ROOT, RANDOM_SEED, OPTUNA_N_TRIALS,
    MODEL_CONFIGS
)
from main import NoShowPipeline

logger = logging.getLogger(__name__)

# All available models
ALL_MODELS = ['randomforest', 'lightgbm', 'mlp', 'deepfm', 'fttransformer', 'tabnet']


class ModelBenchmark:
    """Comprehensive benchmarking system for all no-show prediction models"""
    
    def __init__(self, models: List[str] = None, n_trials: int = OPTUNA_N_TRIALS, 
                 random_seed: int = RANDOM_SEED):
        """
        Initialize benchmark system
        
        Args:
            models: List of models to benchmark (default: all models)
            n_trials: Number of Optuna trials per model
            random_seed: Random seed for reproducibility
        """
        self.models = models or ALL_MODELS
        self.n_trials = n_trials
        self.random_seed = random_seed
        
        # Results storage
        self.results = {}
        self.best_params = {}
        
        # Create results directories
        (RESULTS_ROOT / 'benchmark').mkdir(parents=True, exist_ok=True)
        (RESULTS_ROOT / 'best_params').mkdir(parents=True, exist_ok=True)
        
    def run_benchmark(self, optimize: bool = True, save_results: bool = True) -> Dict:
        """
        Run comprehensive benchmark on all models
        
        Args:
            optimize: Whether to run hyperparameter optimization
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE MODEL BENCHMARK")
        logger.info("="*80)
        logger.info(f"Models: {self.models}")
        logger.info(f"Optimization trials: {self.n_trials}")
        logger.info(f"Random seed: {self.random_seed}")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'models': self.models,
                'n_trials': self.n_trials,
                'random_seed': self.random_seed,
                'optimize': optimize
            },
            'results': {},
            'summary': {}
        }
        
        # Run each model
        for i, model_name in enumerate(self.models, 1):
            logger.info(f"\\n[Model {i}/{len(self.models)}] Running {model_name.upper()}...")
            logger.info("-" * 60)
            
            try:
                # Initialize pipeline for this model
                pipeline = NoShowPipeline(model_type=model_name, random_seed=self.random_seed)
                
                # Run complete pipeline
                model_results = pipeline.run_complete_pipeline(
                    optimize=optimize, 
                    n_trials=self.n_trials
                )
                
                # Store results
                benchmark_results['results'][model_name] = {
                    'metrics': model_results['metrics'],
                    'best_params': model_results['best_params'],
                    'feature_importance': model_results['feature_importance'].to_dict('records')
                }
                
                # Store best parameters
                self.best_params[model_name] = model_results['best_params']
                
                logger.info(f"✅ {model_name.upper()} completed successfully!")
                
            except Exception as e:
                logger.error(f"❌ {model_name.upper()} failed: {str(e)}")
                benchmark_results['results'][model_name] = {
                    'error': str(e),
                    'metrics': {},
                    'best_params': {},
                    'feature_importance': []
                }
        
        # Generate summary
        benchmark_results['summary'] = self._generate_summary(benchmark_results['results'])
        
        # Save results if requested
        if save_results:
            self._save_benchmark_results(benchmark_results)
            self._save_best_parameters()
            self._save_comparison_csv(benchmark_results)
        
        # Print final summary
        self._print_benchmark_summary(benchmark_results['summary'])
        
        return benchmark_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from benchmark results"""
        summary_data = []
        
        for model_name, model_result in results.items():
            if 'error' not in model_result:
                metrics = model_result['metrics']
                
                # Extract key metrics
                summary_data.append({
                    'model': model_name,
                    'roc_auc': metrics.get('roc_auc', 0.0),
                    'pr_auc': metrics.get('pr_auc', 0.0),
                    'f1': metrics.get('f1', 0.0),
                    'f2': metrics.get('f2', 0.0),
                    'mcc': metrics.get('mcc', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'accuracy': metrics.get('accuracy', 0.0),
                    'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                    'train_time': metrics.get('train_time', 0.0),
                    'pred_time_ms': metrics.get('pred_time_ms', 0.0),
                    'brier_score': metrics.get('brier_score', 0.0),
                    'g_mean': metrics.get('g_mean', 0.0)
                })
        
        if not summary_data:
            return {'error': 'No successful model runs'}
        
        # Create summary DataFrame
        df = pd.DataFrame(summary_data)
        
        # Find best models for each metric
        best_models = {}
        key_metrics = ['roc_auc', 'pr_auc', 'f1', 'f2', 'mcc']
        
        for metric in key_metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_models[metric] = {
                    'model': df.loc[best_idx, 'model'],
                    'value': df.loc[best_idx, metric]
                }
        
        return {
            'data': summary_data,
            'best_models': best_models,
            'total_models': len(summary_data),
            'successful_models': len([r for r in results.values() if 'error' not in r])
        }
    
    def _save_benchmark_results(self, results: Dict):
        """Save comprehensive benchmark results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RESULTS_ROOT / 'benchmark' / f'benchmark_results_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def _save_best_parameters(self):
        """Save best parameters for each model to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = RESULTS_ROOT / 'best_params' / f'best_params_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {filepath}")
    
    def _save_comparison_csv(self, results: Dict):
        """Save model comparison results to CSV"""
        if results['summary'].get('data'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = RESULTS_ROOT / 'benchmark' / f'model_comparison_{timestamp}.csv'
            
            df = pd.DataFrame(results['summary']['data'])
            
            # Round numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(4)
            
            # Sort by PR-AUC (primary metric for imbalanced data)
            df = df.sort_values('pr_auc', ascending=False)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Comparison CSV saved to {filepath}")
    
    def _print_benchmark_summary(self, summary: Dict):
        """Print formatted benchmark summary"""
        print("\\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        if 'error' in summary:
            print(f"❌ Benchmark failed: {summary['error']}")
            return
        
        print(f"✅ Successfully completed {summary['successful_models']}/{summary['total_models']} models\\n")
        
        # Create results table
        if summary.get('data'):
            df = pd.DataFrame(summary['data'])
            
            # Select key columns for display
            display_cols = ['model', 'roc_auc', 'pr_auc', 'f1', 'f2', 'mcc', 
                           'train_time', 'pred_time_ms']
            
            # Create display table
            display_df = df[display_cols].copy()
            
            # Format model names
            display_df['model'] = display_df['model'].str.upper()
            
            # Round numeric values
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            # Sort by PR-AUC
            display_df = display_df.sort_values('pr_auc', ascending=False)
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'model': 'Model',
                'roc_auc': 'ROC-AUC',
                'pr_auc': 'PR-AUC',
                'f1': 'F1',
                'f2': 'F2',
                'mcc': 'MCC',
                'train_time': 'Train(s)',
                'pred_time_ms': 'Pred(ms)'
            })
            
            print(display_df.to_string(index=False))
        
        # Print best models
        if summary.get('best_models'):
            print("\\n" + "="*40)
            print("BEST MODELS BY METRIC")
            print("="*40)
            
            for metric, best in summary['best_models'].items():
                print(f"{metric.upper():12}: {best['model'].upper():15} ({best['value']:.4f})")
        
        print("\\n" + "="*80)


def main():
    """Main entry point for benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Benchmark')
    
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                       choices=ALL_MODELS + ['all'],
                       help='Models to benchmark (default: all)')
    
    parser.add_argument('--trials', type=int, default=OPTUNA_N_TRIALS,
                       help='Number of Optuna trials per model')
    
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                       help='Skip hyperparameter optimization')
    
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Handle 'all' keyword
    if args.models == ['all']:
        args.models = ALL_MODELS
    
    # Run benchmark
    benchmark = ModelBenchmark(
        models=args.models,
        n_trials=args.trials,
        random_seed=args.seed
    )
    
    results = benchmark.run_benchmark(optimize=args.optimize)
    
    print(f"\\nBenchmark completed! Results saved to {RESULTS_ROOT}/benchmark/")


if __name__ == "__main__":
    main()