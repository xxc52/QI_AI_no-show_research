"""
Timer utilities for measuring training and prediction times
Essential for research paper performance reporting
"""

import time
import numpy as np
from typing import Tuple, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start_time
        logger.info(f"{self.name} took {self.elapsed_time:.4f} seconds")
        
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.elapsed_time is None:
            if self.start_time is not None:
                return time.perf_counter() - self.start_time
            else:
                return 0.0
        return self.elapsed_time


def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function execution
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (function_result, elapsed_time_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    
    return result, elapsed_time


def measure_training_time(model, X, y) -> float:
    """
    Measure model training time
    
    Args:
        model: Model with fit method
        X: Training features
        y: Training target
        
    Returns:
        Training time in seconds
    """
    start_time = time.perf_counter()
    model.fit(X, y)
    training_time = time.perf_counter() - start_time
    
    logger.info(f"Training completed in {training_time:.4f} seconds")
    return training_time


def measure_prediction_time(model, X, n_runs: int = 10) -> Tuple[float, float, float]:
    """
    Measure model prediction time
    
    Args:
        model: Trained model with predict_proba method
        X: Features to predict
        n_runs: Number of runs for averaging
        
    Returns:
        Tuple of (total_time_seconds, time_per_sample_ms, std_time_ms)
    """
    times = []
    n_samples = len(X)
    
    # Warm-up run
    _ = model.predict_proba(X)
    
    # Measurement runs
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = model.predict_proba(X)
        elapsed_time = time.perf_counter() - start_time
        times.append(elapsed_time)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    time_per_sample = mean_time / n_samples
    
    # Convert to milliseconds for per-sample time
    time_per_sample_ms = time_per_sample * 1000
    std_time_ms = (std_time / n_samples) * 1000
    
    logger.info(f"Prediction time: {mean_time:.4f}s total, {time_per_sample_ms:.4f}ms per sample")
    
    return mean_time, time_per_sample_ms, std_time_ms


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class ProgressTimer:
    """Timer with progress tracking for long operations"""
    
    def __init__(self, total_steps: int, name: str = "Progress"):
        self.total_steps = total_steps
        self.name = name
        self.current_step = 0
        self.start_time = None
        
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        self.current_step = 0
        logger.info(f"{self.name}: Starting {self.total_steps} steps")
        
    def step(self, n: int = 1):
        """Update progress"""
        self.current_step += n
        
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            progress = self.current_step / self.total_steps
            
            if self.current_step < self.total_steps:
                eta = elapsed * (1 - progress) / progress if progress > 0 else 0
                logger.info(f"{self.name}: Step {self.current_step}/{self.total_steps} "
                          f"({progress*100:.1f}%) - ETA: {format_time(eta)}")
            else:
                logger.info(f"{self.name}: Completed in {format_time(elapsed)}")
    
    def get_elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time is not None:
            return time.perf_counter() - self.start_time
        return 0.0