# -*- coding: utf-8 -*-
"""
설정 파일
- 모델 하이퍼파라미터, 기본값 등
"""

# 기본 하이퍼파라미터
DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 1024,
    "epochs": 50,
    "patience": 6,
    "lr": 1e-3,
    "models": ["mlp", "wide_deep", "deepfm", "ftt", "tabnet"]
}

# 모델별 하이퍼파라미터
MODEL_CONFIGS = {
    "mlp": {
        "hidden": [128, 64],
        "dropout": 0.3
    },
    "wide_deep": {
        "emb_dim": 16,
        "deep_hidden": [128, 64],
        "dropout": 0.3
    },
    "deepfm": {
        "emb_dim": 12,
        "deep_hidden": [128, 64],
        "dropout": 0.3
    },
    "ftt": {
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.2
    },
    "tabnet": {
        "cat_emb_dim": 1,
        "lr": 1e-2
    }
}

# 데이터 전처리 설정
PREPROCESSING_CONFIG = {
    "target_col": "No-show",
    "test_size": 0.2,
    "val_size": 0.25,  # train의 25% = 전체의 20%
    "random_state": 42
}

# 학습 설정
TRAINING_CONFIG = {
    "weight_decay": 1e-5,
    "num_workers": 0,
    "virtual_batch_size": 256
} 