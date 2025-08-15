# -*- coding: utf-8 -*-
"""
Wide & Deep 모델 모듈
- 선형 모델(Wide)과 신경망(Deep)의 결합
"""

import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    """
    Wide & Deep 모델
    
    Wide: 선형(원-핫) -> logit
    Deep: [임베딩(범주형) + 연속형] -> MLP -> logit
    최종: wide_logit + deep_logit
    """
    
    def __init__(self, cat_dims, num_dim, wide_dim, emb_dim=16, deep_hidden=[128,64], dropout=0.3):
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
        
        # 임베딩 레이어
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_dim, min(emb_dim, max(4, int(round(cat_dim**0.25))))) 
            for cat_dim in cat_dims
        ])
        emb_total = sum([emb.embedding_dim for emb in self.emb_layers])

        # Deep MLP
        deep_in = emb_total + num_dim
        deep_layers = []
        prev = deep_in
        
        for h in deep_hidden:
            deep_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        
        deep_layers += [nn.Linear(prev, 1)]
        self.deep = nn.Sequential(*deep_layers)

        # Wide (선형)
        self.wide = nn.Linear(wide_dim, 1)

    def forward(self, x_num, x_wide, x_cat):
        """순전파"""
        # 범주형 임베딩 처리
        if len(self.emb_layers) > 0 and x_cat.shape[1] > 0:
            emb_list = []
            for i, emb in enumerate(self.emb_layers):
                emb_list.append(emb(x_cat[:, i]))
            x_emb = torch.cat(emb_list, dim=1)
            deep_in = torch.cat([x_emb, x_num], dim=1)
        else:
            # 범주형 변수가 없는 경우
            deep_in = x_num
            
        deep_logit = self.deep(deep_in).squeeze(1)
        
        # Wide 부분
        if x_wide.shape[1] > 0:
            wide_logit = self.wide(x_wide).squeeze(1)
        else:
            wide_logit = torch.zeros_like(deep_logit)
            
        return deep_logit + wide_logit  # logits 