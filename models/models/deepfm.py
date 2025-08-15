# -*- coding: utf-8 -*-
"""
DeepFM 모델 모듈
- Factorization Machine과 Deep MLP의 결합
"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """
    DeepFM 모델
    
    1. Wide: 선형(원-핫) -> logit
    2. FM: 2차 상호작용 -> logit  
    3. Deep: [임베딩(범주형) + 연속형] -> MLP -> logit
    4. 최종: wide_logit + fm_logit + deep_logit
    """
    
    def __init__(self, cat_dims, num_dim, wide_dim, emb_dim=12, deep_hidden=[128,64], dropout=0.3):
        """
        Args:
            cat_dims: 각 범주형 변수의 차원 리스트
            num_dim: 수치형 변수의 차원
            wide_dim: 원-핫 벡터의 차원
            emb_dim: 임베딩 차원
            deep_hidden: Deep MLP의 은닉층 크기
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self._model_type = "deepfm"
        
        # 1) Wide
        self.wide = nn.Linear(wide_dim, 1) if wide_dim > 0 else None

        # 2) FM용 임베딩 (각 카테고리 컬럼별)
        self.emb_layers = nn.ModuleList([nn.Embedding(cdim, emb_dim) for cdim in cat_dims])
        self.emb_dim = emb_dim

        # 3) Deep
        deep_in = emb_dim * len(cat_dims) + num_dim
        layers = []
        prev = deep_in
        
        for h in deep_hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        
        layers += [nn.Linear(prev, 1)]
        self.deep = nn.Sequential(*layers)

    def forward(self, x_num, x_wide, x_cat):
        """순전파"""
        # 범주형 변수가 없는 경우 처리
        if len(self.emb_layers) == 0 or x_cat.shape[1] == 0:
            # 범주형 변수가 없으면 Deep 부분만 사용
            deep_in = x_num
            deep_logit = self.deep(deep_in).squeeze(1)
            
            # Wide 부분
            wide_logit = self.wide(x_wide).squeeze(1) if self.wide is not None else 0.0
            
            # FM 부분은 0으로
            fm_logit = 0.0
            
            return deep_logit + fm_logit + wide_logit
        
        # embeddings: list [B, emb_dim]
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        E = torch.stack(embs, dim=1)  # [B, K, D]

        # FM 2차 상호작용: (sum v)^2 - sum(v^2)
        sum_v = E.sum(dim=1)          # [B, D]
        sum_v_sq = sum_v * sum_v      # [B, D]
        v_sq = E * E                  # [B, K, D]
        sum_vv_sq = v_sq.sum(dim=1)   # [B, D]
        fm_logit = 0.5 * (sum_v_sq - sum_vv_sq).sum(dim=1)  # [B]

        # Deep
        deep_in = torch.cat([E.view(E.size(0), -1), x_num], dim=1)  # [B, K*D + num]
        deep_logit = self.deep(deep_in).squeeze(1)                  # [B]

        # Wide
        wide_logit = self.wide(x_wide).squeeze(1) if self.wide is not None else 0.0

        return deep_logit + fm_logit + wide_logit  # logits 