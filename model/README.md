# Hospital No-Show Prediction ML Pipeline

A comprehensive machine learning pipeline for predicting patient no-show behavior at hospital appointments. This research-focused implementation supports multiple state-of-the-art models optimized for imbalanced data classification.

## 🎯 Project Overview

This pipeline addresses the healthcare challenge of patient no-shows, which affects resource allocation and healthcare delivery efficiency. Using a real-world Brazilian hospital dataset with 108,296 appointments, we implement 6 different models with rigorous evaluation metrics.

### Key Features

- **6 Model Types**: RandomForest, LightGBM, MLP, DeepFM, FT-Transformer, TabNet
- **Temporal Data Splitting**: Realistic train/validation/test splits preserving time dependencies
- **Imbalanced Data Focus**: PR-AUC optimization with comprehensive metrics
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Research-Grade Evaluation**: 15+ metrics including F2, MCC, G-Mean, Cohen's Kappa
- **Comprehensive Benchmarking**: Automated comparison across all models
- **Production-Ready**: Timing metrics for deployment considerations

## 📊 Dataset Information

- **Size**: 108,296 appointments, 62,299 unique patients
- **Time Period**: 2016-04-29 to 2016-06-08 (41 days)
- **Class Distribution**: 20.11% no-show rate (imbalanced)
- **Features**: 24 selected features from univariate statistical tests
- **Data Split**: Temporal 8:1:1 (train:validation:test)

### Feature Categories

1. **Appointment Timing (6)**: lead_time, is_same_day, registration patterns
2. **Patient History (6)**: Previous no-shows, appointment frequency, regularity
3. **Medical Conditions (3)**: Hypertension, Diabetes, Handicap status
4. **Demographics (2)**: Age, Scholarship status
5. **System (1)**: SMS notification received
6. **Geographic (1)**: Neighborhood cluster (6 clusters from 81 areas)
7. **Weather (4)**: Temperature changes, radiation, temperature range
8. **Temporal (1)**: Season encoding

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv_noshow
# Windows:
venv_noshow\Scripts\activate.bat
# Linux/Mac:
source venv_noshow/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Single model with optimization
python main.py --model randomforest --trials 50

# Quick test without optimization
python main.py --model mlp --no-optimize

# Comprehensive benchmark (all models)
python main.py --model all --trials 100

# Custom configuration
python main.py --model lightgbm --trials 25 --seed 123
```

### Advanced Usage

```bash
# Neural networks with specific parameters
python main.py --model fttransformer --trials 30
python main.py --model tabnet --trials 20
python main.py --model deepfm --trials 40

# Skip optimization for quick testing
python main.py --model all --no-optimize
```

## 🏗️ Architecture Overview

### Modular Design

```
model/
├── config/
│   └── config.py           # Centralized configuration
├── data/
│   └── loader.py           # Data loading and preprocessing
├── models/                 # Model implementations
│   ├── random_forest.py    # Scikit-learn RandomForest
│   ├── lightgbm_model.py   # LightGBM with early stopping
│   ├── mlp_model.py        # Multi-Layer Perceptron
│   ├── deepfm_model.py     # DeepFM for tabular data
│   ├── fttransformer_model.py # Feature Tokenizer + Transformer
│   └── tabnet_model.py     # TabNet with attention
├── training/
│   ├── validator.py        # TimeSeriesSplit cross-validation
│   └── optimizer.py        # Optuna Bayesian optimization
├── evaluation/
│   └── metrics.py          # Comprehensive metrics calculation
├── utils/
│   └── timer.py           # Performance timing utilities
├── main.py                # Main orchestrator
└── benchmark.py           # Comprehensive benchmarking
```

### Key Design Principles

1. **Consistent Interface**: All models implement fit/predict/predict_proba
2. **Temporal Integrity**: TimeSeriesSplit prevents data leakage
3. **Imbalanced Data Focus**: PR-AUC optimization, balanced metrics
4. **Research Reproducibility**: Fixed random seeds, deterministic splits
5. **Performance Monitoring**: Training and prediction timing

## 🤖 Supported Models

### 1. RandomForest (Baseline)

- **Type**: Ensemble method
- **Key Params**: n_estimators=1000, optimizes max_depth, min_samples_leaf, min_samples_split
- **Strengths**: Fast, interpretable, robust to overfitting
- **Use Case**: Baseline and interpretability

### 2. LightGBM (Gradient Boosting)

- **Type**: Gradient boosting
- **Key Params**: n_estimators=2000, early_stopping=150, scale_pos_weight=4
- **Strengths**: High performance, efficient, handles categorical features
- **Use Case**: Production deployment, high accuracy

### 3. MLP (Multi-Layer Perceptron)

- **Type**: Deep neural network
- **Architecture**: Configurable layers (small/medium/large)
- **Features**: BatchNorm, Dropout, Adam optimizer
- **Use Case**: Deep learning baseline

### 4. DeepFM (Factorization Machines + Deep)

- **Type**: Hybrid model
- **Components**: Linear + FM + Deep neural network
- **Features**: Embedding for categorical features, feature interactions
- **Use Case**: Capturing complex feature interactions

### 5. FT-Transformer (Feature Tokenizer + Transformer)

- **Type**: Transformer architecture
- **Components**: Feature tokenization, multi-head attention, CLS token
- **Features**: Self-attention mechanism, positional embeddings
- **Use Case**: State-of-the-art tabular learning

### 6. TabNet (Sequential Attention)

- **Type**: Attention-based tabular model
- **Components**: Feature selection via attention, decision steps
- **Features**: Interpretable attention masks, feature reuse penalty
- **Use Case**: Interpretable deep learning

## 📈 Evaluation Metrics

### Comprehensive Metrics Suite (15+ metrics)

#### Primary Metric

- **PR-AUC**: Precision-Recall AUC (optimal for imbalanced data)

#### Classification Metrics

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Corrected for class imbalance
- **Precision/Recall**: Positive class focus
- **F1-Score**: Harmonic mean of precision/recall
- **F2-Score**: Recall-weighted F-score (clinical focus)

#### Advanced Metrics

- **MCC**: Matthews Correlation Coefficient (balanced measure)
- **Cohen's Kappa**: Agreement correcting for chance
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **G-Mean**: Geometric mean of sensitivity/specificity
- **Brier Score**: Probabilistic accuracy
- **Log Loss**: Cross-entropy loss

#### Business Metrics

- **Top-K Precision/Recall**: Performance at 10%, 20%, 30% thresholds
- **Youden's J**: Optimal threshold identification

#### Performance Metrics

- **Training Time**: Model fitting duration
- **Prediction Time**: Per-sample inference time
- **Memory Usage**: Resource consumption

## ⚙️ Configuration

### Model-Specific Parameters

#### RandomForest (Optimized for Efficiency)

```python
RF_DEFAULT_PARAMS = {
    'n_estimators': 1000,      # Fixed for stability
    'criterion': 'gini',       # Fixed for efficiency
    'max_features': 'sqrt',    # Fixed for efficiency
    'class_weight': 'balanced' # Fixed for imbalanced data
}

RF_SEARCH_SPACE = {
    'max_depth': [None, 15],           # None vs 15
    'min_samples_leaf': [1, 5],        # 1 vs 5
    'min_samples_split': [2, 10]       # Range optimization
}
```

#### LightGBM (Enhanced Configuration)

```python
LGBM_DEFAULT_PARAMS = {
    'n_estimators': 2000,          # Increased for performance
    'learning_rate': 0.05,         # Reduced for stability
    'min_child_samples': 100,      # Increased from 20
    'bagging_freq': 1,             # Updated from 5
    'scale_pos_weight': 4,         # Instead of class_weight
    'metric': 'average_precision'  # PR-AUC alignment
}

LGBM_SEARCH_SPACE = {
    'max_depth': [6, 12],              # Expanded range
    'num_leaves': [31, 127],           # Expanded range
    'min_child_samples': [50, 200],    # Expanded range
    'feature_fraction': [0.6, 0.95],   # Added to search
    'bagging_fraction': [0.6, 0.95],   # Added to search
    'min_split_gain': [0.0, 0.1],      # Added regularization
    'lambda_l1': [0.0, 10.0],          # L1 regularization
    'lambda_l2': [0.0, 10.0]           # L2 regularization
}
```

#### Neural Network Models

- **Early Stopping**: 50 rounds of no improvement
- **Optimizer**: Adam with weight decay
- **Loss Function**: BCEWithLogitsLoss with pos_weight
- **Batch Sizes**: Model-specific (256-2048)
- **Learning Rates**: Model-specific (1e-4 to 2e-2)

### Training Configuration

```python
CV_FOLDS = 5                    # TimeSeriesSplit folds
EARLY_STOPPING_ROUNDS = 150     # Enhanced from 50
OPTUNA_N_TRIALS = 100          # Bayesian optimization trials
RANDOM_SEED = 42               # Reproducibility
```

## 📊 Benchmarking System

### Comprehensive Comparison

```bash
# Run all models with optimization
python main.py --model all --trials 100
```

### Output Files

1. **Benchmark Results**: `results/benchmark/benchmark_results_YYYYMMDD_HHMMSS.json`

   - Complete results with metrics, parameters, feature importance
   - Timestamp for version tracking
   - Error handling for failed models

2. **Best Parameters**: `results/best_params/best_params_YYYYMMDD_HHMMSS.json`

   - Optimal hyperparameters for each model
   - Ready for production deployment
   - Model-specific configurations

3. **Comparison CSV**: `results/benchmark/model_comparison_YYYYMMDD_HHMMSS.csv`
   - Sortable comparison table
   - All metrics for easy analysis
   - Publication-ready format

### Sample Output

```
Model        ROC-AUC  PR-AUC  F1     F2     MCC    Train(s)  Pred(ms)
LIGHTGBM     0.6745   0.3890  0.3654 0.4123 0.2891 12.5      1.23
TABNET       0.6712   0.3845  0.3612 0.4089 0.2847 245.8     3.45
DEEPFM       0.6689   0.3823  0.3598 0.4067 0.2829 189.2     2.78
FTTRANSFORMER 0.6678  0.3812  0.3587 0.4056 0.2818 156.4     2.34
MLP          0.6654   0.3789  0.3568 0.4034 0.2796 89.7      1.89
RANDOMFOREST 0.6642   0.3776  0.3556 0.4023 0.2785 15.2      0.98
```

## 🔄 Data Pipeline

### Temporal Data Loading

```python
# Automatic temporal split detection
data_loader = DataLoader()
X_train, y_train, X_test, y_test, train_ids, test_ids = data_loader.get_train_test_split()
```

### Data Preprocessing

1. **ID Extraction**: Separate PatientId/AppointmentID for tracking
2. **Feature Selection**: Load 24 pre-selected features
3. **Temporal Combination**: Merge train+validation for TimeSeriesSplit
4. **Scaling**: StandardScaler for neural networks
5. **Class Weights**: Automatic imbalance handling

### Validation Strategy

- **TimeSeriesSplit**: 5-fold temporal cross-validation
- **No Data Leakage**: Patient-aware splitting
- **Chronological Order**: Respects temporal dependencies
- **Stratification**: Maintains class distribution

## 🎯 Optimization Strategy

### Bayesian Optimization with Optuna

#### Objective Function

- **Primary Metric**: PR-AUC (best for imbalanced data)
- **Cross-Validation**: 5-fold TimeSeriesSplit
- **Efficiency Focus**: Reduced search spaces
- **Early Stopping**: Prevents overfitting

#### Search Strategy

1. **Tree-structured Parzen Estimator**: Efficient search
2. **Pruning**: Early termination of poor trials
3. **Reproducibility**: Fixed random seeds
4. **Progress Tracking**: Real-time optimization progress

#### Model-Specific Tuning

- **Classical ML**: 3-8 hyperparameters (efficiency-focused)
- **Neural Networks**: Architecture + training hyperparameters
- **Ensemble Methods**: Tree-specific parameters
- **Attention Models**: Architecture-specific tuning

## 📋 Results Interpretation

### Primary Focus: PR-AUC

- **Range**: 0.0 - 1.0 (higher is better)
- **Baseline**: 0.2011 (random classifier for 20.11% positive rate)
- **Good Performance**: > 0.35
- **Excellent Performance**: > 0.40

### Secondary Metrics

- **F2-Score**: Emphasizes recall (clinical importance)
- **MCC**: Balanced measure accounting for all confusion matrix cells
- **G-Mean**: Balance between sensitivity and specificity
- **Cohen's Kappa**: Agreement beyond chance

### Performance Considerations

- **Training Time**: Deployment feasibility
- **Prediction Time**: Real-time application requirements
- **Memory Usage**: Resource constraints
- **Interpretability**: Clinical decision support

### Model Selection Criteria

1. **Primary**: PR-AUC performance
2. **Secondary**: F2-score (clinical focus)
3. **Practical**: Training/prediction time
4. **Stability**: Cross-validation consistency

## 🚀 Production Deployment

### Model Persistence

```python
# Models saved as pickle files
model.save('results/models/lightgbm_20240830_143022.pkl')

# Load for prediction
from models.lightgbm_model import LightGBMNoShow
model = LightGBMNoShow.load('path/to/model.pkl')
predictions = model.predict(new_data)
```

### Performance Monitoring

- **Training Time**: Model fitting duration
- **Prediction Time**: Per-sample inference (ms)
- **Batch Prediction**: Throughput optimization
- **Memory Footprint**: Resource planning

### Deployment Checklist

- [ ] Model performance validation (PR-AUC > threshold)
- [ ] Prediction time requirements met
- [ ] Memory constraints satisfied
- [ ] Feature preprocessing pipeline tested
- [ ] Error handling implemented
- [ ] Monitoring and logging configured

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Missing dependencies
pip install lightgbm pytorch-tabnet

# Path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Memory Issues

```python
# Reduce batch size for neural networks
python main.py --model mlp --batch_size 256

# Use fewer Optuna trials
python main.py --model all --trials 50
```

#### 3. Performance Issues

```bash
# Skip optimization for quick testing
python main.py --model randomforest --no-optimize

# Single model testing
python main.py --model mlp --trials 1
```

#### 4. CUDA Issues

```python
# Force CPU usage
import torch
torch.device('cpu')  # Models auto-detect device
```

### Model-Specific Issues

#### TabNet Dimension Errors ✅ FIXED

- **Issue**: Attention mechanism dimension mismatch
- **Solution**: Proper M_feature/M_att splitting in forward pass
- **Status**: Resolved - TabNet now works correctly

#### LightGBM Metric Mismatch ✅ FIXED

- **Issue**: Optuna used PR-AUC but LightGBM used binary_logloss
- **Solution**: Aligned both to use 'average_precision' (PR-AUC)
- **Status**: Resolved - Consistent optimization

#### Neural Network Convergence

- **Solution**: Reduce learning rate, increase patience
- **Monitoring**: Early stopping prevents overfitting

## 🏆 Recent Updates

### Version 1.0 (August 2024)

- ✅ **TabNet Fixed**: Resolved dimension mismatch in attention mechanism
- ✅ **Enhanced LightGBM**: Updated to scale_pos_weight=4, expanded search space
- ✅ **RandomForest Optimized**: Reduced to 3 core hyperparameters for efficiency
- ✅ **All 6 Models Working**: RandomForest, LightGBM, MLP, DeepFM, FT-Transformer, TabNet
- ✅ **Comprehensive Benchmarking**: Full comparison system with --model all
- ✅ **Production Ready**: Timing metrics and deployment considerations

### Configuration Updates

- **Early Stopping**: Increased to 150 rounds (from 50)
- **LightGBM**: n_estimators=2000, learning_rate=0.05, min_child_samples=100
- **RandomForest**: Focused optimization on max_depth [None, 15], min_samples_leaf [1, 5]
- **Search Spaces**: Expanded LightGBM with 8 parameters, reduced RandomForest to 3

## 📚 References & Citations

### Academic Background

- **Original Dataset**: Alissom Aquino, Vitória Hospital, Brazil
- **TabNet**: "TabNet: Attentive Interpretable Tabular Learning" (Arık et al., 2019)
- **FT-Transformer**: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- **DeepFM**: "DeepFM: A Factorization-Machine based Neural Network" (Guo et al., 2017)

### Implementation Libraries

- **LightGBM**: Microsoft's gradient boosting framework
- **PyTorch**: Deep learning models (MLP, DeepFM, FT-Transformer, TabNet)
- **Scikit-learn**: RandomForest and preprocessing utilities
- **Optuna**: Bayesian hyperparameter optimization

### Research Context

This pipeline was developed for QI AI research on hospital no-show prediction, focusing on:

- **Clinical Relevance**: F2-score emphasis on recall
- **Real-world Deployment**: Timing and resource considerations
- **Interpretability**: Attention-based models for clinical decision support
- **Reproducibility**: Fixed seeds and deterministic processes

## 📞 Support & Contact

For questions, issues, or contributions:

- **GitHub Issues**: Repository issue tracker
- **Documentation**: This README and inline code comments
- **Research Context**: See main project CLAUDE.md

---

**Last Updated**: August 2024  
**Version**: 1.0.0  
**Status**: Production Ready ✅  
**All Models**: Working and Benchmarked ✅
