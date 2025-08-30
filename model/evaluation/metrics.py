"""
Comprehensive metrics for imbalanced classification
Research-focused evaluation for no-show prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, confusion_matrix,
    brier_score_loss, log_loss, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import label_binarize
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for imbalanced classification
    Includes both standard and imbalanced-specific metrics
    """
    
    def __init__(self, pos_label: int = 1):
        """
        Initialize metrics calculator
        
        Args:
            pos_label: Positive class label (1 for no-show)
        """
        self.pos_label = pos_label
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None,
                            include_curves: bool = False) -> Dict:
        """
        Calculate comprehensive metrics for paper results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for probabilistic metrics)
            include_curves: Whether to include ROC and PR curves
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Imbalanced data metrics
        metrics.update(self._calculate_imbalanced_metrics(y_true, y_pred))
        
        # Probabilistic metrics (if probabilities provided)
        if y_prob is not None:
            metrics.update(self._calculate_probabilistic_metrics(y_true, y_prob))
            metrics.update(self._calculate_threshold_metrics(y_true, y_prob))
            metrics.update(self._calculate_top_k_metrics(y_true, y_prob))
            
            if include_curves:
                metrics.update(self._calculate_curves(y_true, y_prob))
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate standard classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
        }
    
    def _calculate_imbalanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate metrics specifically for imbalanced data"""
        return {
            'f2': self._calculate_fbeta(y_true, y_pred, beta=2.0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'g_mean': self._calculate_g_mean(y_true, y_pred),
        }
    
    def _calculate_probabilistic_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate probability-based metrics"""
        metrics = {}
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = np.nan
            
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except:
            metrics['pr_auc'] = np.nan
            
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        metrics['log_loss'] = log_loss(y_true, y_prob)
        
        return metrics
    
    def _calculate_threshold_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate threshold-dependent metrics"""
        # Find optimal threshold based on F2 score
        optimal_threshold, best_f2 = self.find_optimal_threshold(y_true, y_prob, metric='f2')
        
        # Calculate Youden's J statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = np.max(tpr - fpr)
        
        return {
            'optimal_threshold': optimal_threshold,
            'best_f2_score': best_f2,
            'youdens_j': youden_j
        }
    
    def _calculate_top_k_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                k_values: List[int] = [10, 20, 30]) -> Dict:
        """
        Calculate Precision and Recall at top K%
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k_values: List of K percentages to evaluate
            
        Returns:
            Dictionary with top-k metrics
        """
        metrics = {}
        n_samples = len(y_true)
        
        # Sort by predicted probability
        sorted_indices = np.argsort(y_prob)[::-1]
        
        for k in k_values:
            # Get top k% of predictions
            k_samples = int(n_samples * k / 100)
            top_k_indices = sorted_indices[:k_samples]
            
            # Create predictions (1 for top k%, 0 for rest)
            y_pred_k = np.zeros_like(y_true)
            y_pred_k[top_k_indices] = 1
            
            # Calculate metrics
            precision_k = precision_score(y_true, y_pred_k, zero_division=0)
            recall_k = recall_score(y_true, y_pred_k, zero_division=0)
            
            metrics[f'precision_at_{k}'] = precision_k
            metrics[f'recall_at_{k}'] = recall_k
        
        return metrics
    
    def _calculate_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate ROC and PR curves"""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        
        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        return {
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_g_mean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate geometric mean of sensitivity and specificity"""
        sensitivity = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return np.sqrt(sensitivity * specificity)
    
    def _calculate_fbeta(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
        """
        Calculate F-beta score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            beta: Beta parameter (2 for F2 score)
            
        Returns:
            F-beta score
        """
        precision = precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        
        if precision + recall == 0:
            return 0
        
        beta_squared = beta ** 2
        return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              metric: str = 'f2') -> Tuple[float, float]:
        """
        Find optimal threshold for given metric
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'f2', 'g_mean', 'youden')
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
            elif metric == 'f2':
                score = self._calculate_fbeta(y_true, y_pred, beta=2.0)
            elif metric == 'g_mean':
                score = self._calculate_g_mean(y_true, y_pred)
            elif metric == 'youden':
                sensitivity = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
                specificity = self._calculate_specificity(y_true, y_pred)
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def format_metrics_table(self, metrics: Dict, model_name: str = "Model") -> str:
        """
        Format metrics as a nice table for research paper
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
            
        Returns:
            Formatted string table
        """
        table = f"\n{model_name} - Performance Metrics\n"
        table += "=" * 60 + "\n"
        
        # Basic metrics
        table += "\nBasic Metrics:\n"
        table += f"  Accuracy:           {metrics.get('accuracy', 0):.4f}\n"
        table += f"  Precision:          {metrics.get('precision', 0):.4f}\n"
        table += f"  Recall:             {metrics.get('recall', 0):.4f}\n"
        table += f"  Specificity:        {metrics.get('specificity', 0):.4f}\n"
        table += f"  F1-Score:           {metrics.get('f1', 0):.4f}\n"
        
        # Imbalanced metrics
        table += "\nImbalanced Data Metrics:\n"
        table += f"  F2-Score:           {metrics.get('f2', 0):.4f}\n"
        table += f"  MCC:                {metrics.get('mcc', 0):.4f}\n"
        table += f"  Cohen's Kappa:      {metrics.get('cohen_kappa', 0):.4f}\n"
        table += f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}\n"
        table += f"  G-Mean:             {metrics.get('g_mean', 0):.4f}\n"
        
        # Probabilistic metrics
        if 'roc_auc' in metrics:
            table += "\nProbabilistic Metrics:\n"
            table += f"  ROC-AUC:            {metrics.get('roc_auc', 0):.4f}\n"
            table += f"  PR-AUC:             {metrics.get('pr_auc', 0):.4f}\n"
            table += f"  Brier Score:        {metrics.get('brier_score', 0):.4f}\n"
            table += f"  Log Loss:           {metrics.get('log_loss', 0):.4f}\n"
        
        # Threshold metrics
        if 'optimal_threshold' in metrics:
            table += "\nOptimal Threshold:\n"
            table += f"  Threshold:          {metrics.get('optimal_threshold', 0.5):.3f}\n"
            table += f"  Best F2-Score:      {metrics.get('best_f2_score', 0):.4f}\n"
            table += f"  Youden's J:         {metrics.get('youdens_j', 0):.4f}\n"
        
        # Top-K metrics
        if 'precision_at_10' in metrics:
            table += "\nTop-K Performance:\n"
            for k in [10, 20, 30]:
                if f'precision_at_{k}' in metrics:
                    table += f"  Precision@{k}%:      {metrics.get(f'precision_at_{k}', 0):.4f}\n"
                    table += f"  Recall@{k}%:         {metrics.get(f'recall_at_{k}', 0):.4f}\n"
        
        # Timing metrics
        if 'train_time' in metrics:
            table += "\nEfficiency Metrics:\n"
            table += f"  Training Time:      {metrics.get('train_time', 0):.2f} seconds\n"
            table += f"  Prediction Time:    {metrics.get('pred_time_ms', 0):.2f} ms/sample\n"
        
        table += "=" * 60 + "\n"
        
        return table