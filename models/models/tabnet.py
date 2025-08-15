# -*- coding: utf-8 -*-
"""
TabNet 모델 모듈
- pytorch-tabnet 기반 모델 (선택적)
"""

import numpy as np
import torch

# TabNet은 선택적(설치 시에만 사용)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class TabNetWrapper:
    """TabNet 모델 래퍼 클래스"""
    
    def __init__(self, cat_idxs, cat_dims, cat_emb_dim=1, device="cpu"):
        """
        Args:
            cat_idxs: 범주형 변수의 인덱스 리스트
            cat_dims: 각 범주형 변수의 차원 리스트
            cat_emb_dim: 범주형 임베딩 차원
            device: 디바이스
        """
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet이 설치되지 않았습니다. pip install pytorch-tabnet")
        
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.device = device
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=2048, lr=1e-2):
        """모델 학습"""
        self.model = TabNetClassifier(
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_params={"step_size": 20, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=1,
            device_name=self.device
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["auc_pr"],
            max_epochs=epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )
        
        return self
        
    def predict_proba(self, X):
        """확률 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """클래스 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        return self.model.predict(X) 