# GPU Setup Guide for No-Show Prediction Models

## Overview
This project supports GPU acceleration for both traditional ML (LightGBM) and deep learning models (MLP, DeepFM, FT-Transformer, TabNet).

## Neural Network Models (PyTorch-based)

All neural network models **automatically detect and use GPU** if available:
- MLP
- DeepFM  
- FT-Transformer
- TabNet

### Automatic GPU Detection
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The models will automatically:
1. Check if CUDA is available
2. Use GPU if available
3. Fall back to CPU if not available
4. Log the device being used during training

### Requirements for GPU Usage
- NVIDIA GPU with CUDA support
- PyTorch with CUDA support installed:
  ```bash
  # For CUDA 11.8
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  # For CUDA 12.1
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

## LightGBM GPU Setup

LightGBM requires additional setup for GPU support:

### Requirements
1. NVIDIA GPU with OpenCL support
2. LightGBM compiled with GPU support:
   ```bash
   pip install lightgbm --install-option=--gpu
   ```
   Or build from source following [official GPU tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)

### Enabling GPU for LightGBM
Set the environment variable before running:
```bash
# Linux/Mac
export LIGHTGBM_USE_GPU=1
python model/main.py --model lightgbm --trials 50

# Windows Command Prompt
set LIGHTGBM_USE_GPU=1
python model/main.py --model lightgbm --trials 50

# Windows PowerShell
$env:LIGHTGBM_USE_GPU="1"
python model/main.py --model lightgbm --trials 50
```

### GPU Status Messages
- **CPU mode**: "LightGBM using CPU mode (set LIGHTGBM_USE_GPU=1 for GPU)"
- **GPU mode**: "LightGBM GPU mode enabled (LIGHTGBM_USE_GPU=1)"

## Verification

### Check PyTorch GPU Availability
```python
import torch
print(f"PyTorch GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
```

### Check During Model Training
All models will log their device during training:
- Neural networks: "Device: cuda" or "Device: cpu"
- LightGBM: GPU/CPU mode message in logs

## Performance Expectations

### With GPU (NVIDIA RTX 3090 or similar):
- **LightGBM**: 2-3x faster training
- **Neural Networks**: 5-10x faster training
- **Batch processing**: Significant speedup for large batch sizes

### Without GPU (CPU only):
- **RandomForest**: ~7-10 seconds per trial
- **LightGBM**: ~10-15 seconds per trial
- **MLP**: ~2-3 minutes per trial
- **DeepFM**: ~3-4 minutes per trial
- **FT-Transformer**: ~4-5 minutes per trial
- **TabNet**: ~3-4 minutes per trial

## Troubleshooting

### Neural Networks Not Using GPU
1. Check PyTorch installation:
   ```python
   import torch
   print(torch.version.cuda)  # Should show CUDA version
   ```

2. Reinstall PyTorch with CUDA support (see Requirements above)

### LightGBM GPU Issues
1. Common error: "GPU Tree Learner was not enabled in this build"
   - Solution: Reinstall LightGBM with GPU support
   
2. Segmentation fault with GPU
   - Solution: Use CPU mode (don't set LIGHTGBM_USE_GPU)
   - Some parameter combinations may not work with GPU

### Memory Issues
- Reduce batch size for neural networks
- Reduce `num_leaves` for LightGBM
- Use gradient accumulation for large models

## Recommended Settings

### For Development/Testing (CPU):
```bash
# Quick test with default parameters
python model/main.py --model lightgbm --no-optimize

# Small optimization
python model/main.py --model randomforest --trials 5
```

### For Research/Production (GPU):
```bash
# Full optimization with GPU
export LIGHTGBM_USE_GPU=1  # For LightGBM
python model/main.py --model all --trials 100
```

## Notes
- GPU acceleration is most beneficial for:
  - Large datasets (>100k samples)
  - Deep neural networks
  - Hyperparameter optimization with many trials
- RandomForest does not support GPU (uses parallel CPU cores instead)
- Always monitor GPU memory usage to avoid out-of-memory errors